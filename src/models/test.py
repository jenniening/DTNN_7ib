import os
import tensorflow as tf
import numpy as np
import sys
import logging
from ase.units import kcal, mol

import read_tfrecord
from readtfrecord import read_tfrecord
import model
from model import get_model


def evalmodel(refidx, test_position, test_data, test_size, test_type,
              batch_size,model_name, model_dir,
              transfer_learning, mu, std, atom_ref,logger):

    args = model_name.split("_")[1:]
    n_basis = int(args[9:10])
    n_interactions = int(args[10:11])
    cutoff = float(args[11:12])
    step = float(args[12:13])
    activation = args[13:14]
    reuse = args[14:15]

    #testtf = dbdir + 'test.tfrecord'
    #testlive = dbdir + 'test_live.tfrecord'
#
    #model_dir = '../model'


    # Get Test Data shape

    #test_size1 = sum(1 for exmaple in tf.python_io.tf_record_iterator(testtf))
    #test_size2 = sum(1 for exmaple in tf.python_io.tf_record_iterator(testlive))
    ##test_size = test_size1 + test_size2
    #test_iteration = np.floor_divide(test_size,100)
    #test_final = test_size - test_iteration * 100
    #print("Total test:" + str(test_size))
    ##print(test_final)
    #print(test_iteration)
    test_iteration = np.floor_divide(test_size, batch_size) + 1


    iter_test,test_features  = read_tfrecord(test_data, test_position, batch_size = batch_size, num_epochs= None, shuffle= True,buffer_size = train_size, cutoff = cutoff, step = step)
    test_output = get_model(transfer_learning, test_features, mu, std, atom_ref,reuse,n_basis,n_interactions,activation)
    
    cost = tf.reduce_mean((test_features['targets'][:,refidx]-test_output['y'])**2)
    mae = tf.reduce_mean(tf.abs(test_features['targets'][:,refidx]-test_output['y']))

    with tf.Session() as sess:
        sess.run(iter_test.initializer)
        chkpt_saver = tf.train.Saver()
        checkpoint_path = os.path.join(model_dir,model_name)
        if test_type  == "val":
            chkfile = open(checkpoint_path + "/validation.txt")
            chkpt = chkfile.readline().split('"')[1].strip('"\n')
            chkpt = os.path.join(checkpoint_path, chkpt)
        elif test_type == "total":
            chkfile = open(checkpoint_path + "/total.txt")
            chkpt = chkfile.readline().split('"')[1].strip('"\n')
            chkpt = os.path.join(checkpoint_path, chkpt)
        else:
            chkpt = tf.train.latest_checkpoint(checkpoint_path)
        print(chkpt)
        chkpt_saver.restore(sess, chkpt)

        testrmse = []
        testmae = []
        for i in range(0, test_iteration ):
            print(i,test_iteration)
            trmse, tmae = sess.run([cost, mae])

            testrmse.append(trmse)
            testmae.append(tmae)
        print(len(testmae))
        logger.info("Test Set")
        logger.info("RMSE, MAE")
        if refidx in [3,4,8,14]:
            logger.info('{:8.4f} {:8.4f}\n'.format(np.sqrt(np.mean(testrmse)), np.mean(testmae)))
        else:
            logger.info('{:8.4f} {:8.4f}\n'.format(np.sqrt(np.mean(testrmse))*1./(kcal/mol), np.mean(testmae)*1./(kcal/mol)))

    return None
