### model training ###
import os
import tensorflow as tf
import numpy as np
from readtfrecord import read_tfrecord
import model
from model import Model

import sys
from ase.units import kcal, mol
from datetime import datetime
import logging

###### Getting Hyperparameters ######

def get_optimizer(opt,lr):
    ''' Get Optimizer
    :param opt: optimizer type
    :param lr: learning rate
    '''
    if opt == "Adam":
        optimizer = tf.train.AdamOptimizer(lr)
    elif opt == "GD":
        optimizer = tf.train.GradientDescentOptimizer(lr)

    return optimizer

def get_model(transferlearning, train_features,
                mu, std, atom_ref,
                reuse, n_basis, n_interactions, 
                activation):
    ''' Get Model 

    :param reuse: share or update weights in interaction block
    :type reuse: str
    :param train_features: input features
    :type train_features: tensor
    :param n_basis: length of atom embedding
    :type n_basis: int
    :param n_interactions: depth of interaction blcok
    :type n_interactions: int
    :param activation: activation function
    :type activation: str
    :param mu: mu
    :param std: std
    :param atom_ref: reference
    :return: model information (output)
    '''

    if reuse == "update":
        if transferlearning:
            model = Model(train_features,n_basis,n_interactions, activation, mu, std, atom_ref,2,transferlearning)
        else:
            model = Model(train_features,n_basis,n_interactions, activation, mu, std, atom_ref,0,transferlearning)
    else:
        print("Weights should be updated in interaction block")
    return model.mymodel()

def get_lr(lr_int, i, cycle_steps):
    '''Get current learning rate in cyclic learning
    
    :param lr_int: initial learning rate
    :type lr_int: float
    :param i: step
    :type i: int
    :param cycle_steps: total steps in one cycle
    :type cycle_steps: int
    :return: learning rate
    :rtype: float
    '''

    if i == 0:
        lr = lr_int
    else:
        lr = lr_int/2 * (np.cos(np.pi * np.mod(i-1,np.ceil(cycle_steps))/np.ceil(cycle_steps)) + 1)

    return lr
    
def get_cyclic_learning(cycles, n_iterations):
    '''Define steps in each cycle
    
    :param cycles: number of cycles
    :type cycles: int
    :param n_iterations: total iterations of training
    :type n_iterations: int
    :return: number of steps in each cycle
    :rtype: int
    '''
    cycle_steps = n_iterations/cycles

    return cycle_steps


def get_data(idx):
    '''Get input data in small to large training 
    
    '''
    dbdir = ""
    traintf = dbdir + 'train_nolarger6_' + str(idx) + '.tfrecord'
    #traintf = dbdir + 'train.tfrecord' 
    #val_nolarger6_nocross.tfrecord is same as val_7.tfrecord
    validtf = dbdir + 'val_nolarger6_' + str(idx) + '.tfrecord'
    #validtf = dbdir + 'validation.tfrecord'
    testtf = dbdir + 'val_7.tfrecord'
    #testtf = dbdir + 'test_live.tfrecord'

    return traintf, validtf, testtf


def transfer_learning_scope(num_readout_new = 2):
    '''Get transfer learning retrain scope and restore scope
    
    :param num_readout_new: [description], defaults to 2
    :param num_readout_new: int, optional
    :return: retrain scope and restore scope
    :rtype: str
    '''

    scope_train = "Dense_2|Dense_3|"
    for i in range(num_readout_new):
        scope_train += (i+1) * "new_" + "layer|"
    
    scope_restore = "Dense/W|Dense_1|in2fac|fac2out"

    return scope_train, scope_restore

def read_mustd(infile,refidx):
    ''' Get mu, std
    
    :param infile: file with mu, std information
    :param refidx: target idx
    '''
    for line in open(infile):
        if int(line.split(",")[0]) == int(refidx):
            mu = np.array([float(line.split(",")[1])])
            std = np.array([float(line.split(",")[2].strip("\n"))])
    return mu, std

def get_ref(infile, refidx):
    ''' Get atom reference
    
    :param infile: file with reference information
    :refidx: target idx
    '''
    refidx = int(refidx)
    if refidx == 9:
        ref = 0
    elif refidx == 10:
        ref = 1
    elif refidx == 11:
        ref = 2
    elif refidx == 12:
        ref = 3
    elif refidx == 13:
        ref = 4
    elif refidx == 14:
        ref = 5
    else:
        ref = 0
    reference = np.load(infile)
    atom_ref = reference['atom_ref'][:, ref:(ref+1)]
    return atom_ref


###### Training Models ######

def runmodel(refidx, train_position, train_data, val_data, test_data, train_size, val_size, test_size, 
            batch_size, n_basis, n_interactions, cutoff, step, activation, reuse, opt, train_schedule, cycles, lr_int, decay_rate,n_iterations, mu, std, atom_ref, 
            model_name, transfer_learning, restore_model, logger, checkpoint_interval, model_dir):
    ''' Training moel
    
    :param refidx: target idx
    :type refidx: int
    :param train_position: geometries
    :type train_position: str
    :param train_data: training set name
    :type train_data: str
    :param val_data: validation set name
    :type val_data: str
    :param test_data: test_live set name
    :type test_data: str
    :param train_size: size of training set (used as buffer number, can be smaller)
    :type train_size: int
    :param val_size: size of validation set
    :type val_size: int
    :param test_size: size of test_live set
    :type test_size: int
    :param batch_size: batch_size
    :type batch_size: int
    :param n_basis: length of atom embedding
    :type n_basis: int
    :param n_interactions: depth of interaction block
    :type n_interactions: int
    :param cutoff: distance cutoff
    :type cutoff: float
    :param step: step between Gaussians
    :type step: float
    :param activation: activation function type
    :type activation: str
    :param reuse: share or update weights in interaction block
    :type reuse: str
    :param opt: optimizer type
    :type opt: str
    :param train_schedule: training schedule
    :type train_schedule: str
    :param cycles: number of cycles in cyclic training
    :type cycles: int
    :param lr_int: initial learning rate
    :type lr_int: float
    :param decay_rate: learning rate decay rate
    :type decay_rate: float
    :param n_iterations: number of total iterations
    :type n_iterations: int
    :param mu: mu
    :param std: std 
    :param atom_ref: reference
    :param model_name: saving model name
    :type model_name: str
    :param transfer_learning: transfer learning 
    :type transfer_learning: bool
    :param restore_model: restoring model name
    :type restore_model: str
    '''

    ### read data ###
    iter_train,train_features = read_tfrecord(train_data, train_position, batch_size = batch_size, num_epochs= None, shuffle= True,buffer_size = train_size, cutoff = cutoff, step = step)
    iter_valid,valid_features = read_tfrecord(val_data, train_position, batch_size = val_size, num_epochs= None, shuffle = False,cutoff = cutoff, step = step)
    iter_test,test_features = read_tfrecord(test_data, train_position, batch_size = test_size, num_epochs= None, shuffle = False,cutoff = cutoff, step = step)
    ### construct model ###
    train_output = get_model(transfer_learning, train_features,mu,std,atom_ref,reuse,n_basis,n_interactions,activation)
    ### evaluate performance ###
    cost = tf.reduce_mean((train_features['targets'][:,refidx]-train_output['y'])**2)
    mae = tf.reduce_mean(tf.abs(train_features['targets'][:,refidx]-train_output['y']))
    ### learning rate ###
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if train_schedule == "cyclic" or train_schedule == "constant":
        lr = tf.placeholder(tf.float32, shape = [])
    elif train_schedule == "decay":
        lr = tf.train.exponential_decay(lr_int, global_step, int(train_size/batch_size), decay_rate)
    ### optimizer ###
    optimizer = get_optimizer(opt,lr)
    ### transfer learning ###
    if transfer_learning:
        ### determine which variables needed to be trained ###
        scope_train, scope_restore = transfer_learning_scope()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = scope_train)
        train_op = optimizer.minimize(cost, global_step=global_step,var_list=train_vars)
    else:
        train_op = optimizer.minimize(cost, global_step=global_step)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    ### training ###
    with tf.Session() as sess:
        ### initialize dataset ###
        sess.run(iter_train.initializer)
        sess.run(init_op)
        chkpt_saver = tf.train.Saver()
        if train_schedule == "cyclic":
            cycles = cycles
            cycle_steps = get_cyclic_learning(cycles, n_iterations)
            chkpt_saver_snap = tf.train.Saver(max_to_keep = cycles)
        chkpt_saver_val = tf.train.Saver(max_to_keep = 10)
        chkpt_saver_test = tf.train.Saver(max_to_keep = 10)
        chkpt_saver_total = tf.train.Saver(max_to_keep = 10)
        checkpoint_path = os.path.join(model_dir, model_name)
        ### transfer learning ###
        if transfer_learning:
            ### determine which variables needed to be restored ###
            restore_path = restore_model
            restorefile = open(restore_path + "/validation.txt")
            restore = restorefile.readline().split('"')[1].strip('"\n')
            restore = os.path.join(restore_path, restore)
            print("restore:" + str(restore))
            reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope= scope_restore)
            reuse_vars_dict = dict([(var.op.name,var)for var in reuse_vars])
            restore_saver = tf.train.Saver(reuse_vars_dict)
            restore_saver.restore(sess, restore)

        ### check model directory ###
        if not os.path.exists(checkpoint_path):
            start_iter = 0
            os.makedirs(checkpoint_path)
        else:
            ### restore model ###
            chkpt = tf.train.latest_checkpoint(checkpoint_path)
            chkpt_saver.restore(sess,chkpt)
            start_iter = int(chkpt.split('-')[-1])

        print("start_check:" + str(start_iter))
        ### define chkpt directory ###
        chkpt = os.path.join(checkpoint_path, 'checkpoint_1')
        if train_schedule == "cyclic":
            chkpt_snap = os.path.join(checkpoint_path, 'checkpoint_1_snap')
        chkpt_val = os.path.join(checkpoint_path, 'checkpoint_1_val')
        chkpt_test = os.path.join(checkpoint_path, 'checkpoint_1_test')
        chkpt_total = os.path.join(checkpoint_path, 'checkpoint_1_total')

        global_step.assign(start_iter).eval()
        logger.info('\n')
        logger.info("""Time                     step  RMSE(Train, Valid, Test)   MAE(Train, Valid, Test)""")
        value = {"validation":1000,"test":1000, "total":1000}

        for i in range(start_iter, n_iterations+1):
            if train_schedule == "cyclic":
                c_lr= get_lr(lr_int, i, cycle_steps)
                _, rmse_val, mae_val = sess.run([train_op, cost, mae],feed_dict = {lr:c_lr})
            elif train_schedule == "constant":
                c_lr = lr_int
                _, rmse_val, mae_val = sess.run([train_op, cost, mae],feed_dict = {lr:c_lr})
           
            elif train_schedule == "decay":
                _, rmse_val, mae_val = sess.run([train_op, cost, mae])

            ### check validation and test_live performance and save model ###
            if  train_schedule == "cyclic":
                if i % cycle_steps  == 0:
                    chkpt_saver_snap.save(sess, chkpt_snap, i)
            if i % checkpoint_interval == 0:
                chkpt_saver.save(sess, chkpt, i)
                vrmsetmp = []
                vmaetmp = []
                trmsetmp = []
                tmaetmp = []
                sess.run(iter_valid.initializer)
                sess.run(iter_test.initializer)

                valid_features_val = sess.run(valid_features)
                test_features_val = sess.run(test_features)
                vrmse, vmae = sess.run([cost, mae], feed_dict = {train_features['elements']: valid_features_val['elements'],
                                                                train_features['rdf']: valid_features_val['rdf'],
                                                                train_features['targets']: valid_features_val['targets'] })
                trmse, tmae = sess.run([cost, mae], feed_dict = {train_features['elements']: test_features_val['elements'],
                                                                train_features['rdf']: test_features_val['rdf'],
                                                                train_features['targets']: test_features_val['targets'] })

                vrmsetmp.append(vrmse)
                vmaetmp.append(vmae)
                trmsetmp.append(trmse)
                tmaetmp.append(tmae)

                tmp = '  {:8d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(i, np.sqrt(rmse_val)* 1. /(kcal/mol),
                                      np.sqrt(np.mean(vrmsetmp))* 1. /(kcal/mol),
                                      np.sqrt(np.mean(trmsetmp))*  1. /(kcal/mol),
                                      mae_val * 1. /(kcal/mol),
                                      np.mean(vmaetmp) * 1. /(kcal/mol),
                                      np.mean(tmaetmp) * 1. /(kcal/mol)
                                      )
                if np.mean(vmaetmp) < value["validation"]:
                    value["validation"] = np.mean(vmaetmp)
                    chkpt_saver_val.save(sess, chkpt_val, i,latest_filename = "validation.txt")
                    val_min = i
                if np.mean(tmaetmp) < value["test"]:
                    value["test"] = np.mean(tmaetmp)
                    chkpt_saver_test.save(sess, chkpt_test, i,latest_filename = "test.txt")
                    test_min = i
                if np.mean(tmaetmp) + np.mean(vmaetmp) < value["total"]:
                    value["total"] = np.mean(tmaetmp) + np.mean(vmaetmp)
                    chkpt_saver_total.save(sess, chkpt_total, i,latest_filename = "total.txt")
                    total_min = i
                tmp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + tmp
                logger.info(tmp)
                
        logger.info('\n')
        logger.info('validation_min:' + str(val_min) +',test_min:' + str(test_min) + ',total_min:' + str(total_min))
        chkpt_saver.save(sess, chkpt, i)
   