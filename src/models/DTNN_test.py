import os
import tensorflow as tf
import numpy as np
from DTNN_readtfrecord_mmff import read_tfrecord
import DTNN_model
from DTNN_model import Model
import sys
from datetime import datetime
import logging
from ase.units import kcal, mol
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evalmodel(index, refidx, train_position, test_type, mu, std, atom_ref):
    if train_position == "positions":
        method = "QM"
    else:
        method = "MMFF"

    dbdir = '../prepare_data/db/split_' + str(index) + '/'

    testtf = dbdir + 'test.tfrecord'
    testlive = dbdir + 'test_live.tfrecord'

    model_dir = '../model'


    # Get Test Data shape

    test_size1 = sum(1 for exmaple in tf.python_io.tf_record_iterator(testtf))
    test_size2 = sum(1 for exmaple in tf.python_io.tf_record_iterator(testlive))
    test_size = test_size1 + test_size2
    test_iteration = np.floor_divide(test_size,100)
    test_final = test_size - test_iteration * 100
    print("Total test:" + str(test_size))
    #print(test_final)
    #print(test_iteration)


    iter_test,test_features  = read_tfrecord([testtf,testlive], train_position, batch_size = 100, num_epochs = None, shuffle = False)
    model = Model(test_features,mu,std, atom_ref)
    test_output = model.mymodel()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cost = tf.reduce_mean((test_features['targets'][:,refidx]-test_output['y'])**2)
    mae = tf.reduce_mean(tf.abs(test_features['targets'][:,refidx]-test_output['y']))

    cost_final = tf.reduce_mean((test_features['targets'][0:test_final,refidx]-test_output['y'][0:test_final])**2)
    mae_final = tf.reduce_mean(tf.abs(test_features['targets'][0:test_final,refidx]-test_output['y'][0:test_final]))


    with tf.Session() as sess:
        sess.run(iter_test.initializer)
        chkpt_saver = tf.train.Saver()
        checkpoint_path = os.path.join(model_dir, 'model_newmolecule_' +  str(index) + '_' + method + '_' + str(refidx))
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
            print(len(testrmse))

        for i in range(test_iteration, test_iteration + 1):
            print(i,test_iteration)
            trmse, tmae = sess.run([cost_final, mae_final])

            testrmse.append(trmse)
            testmae.append(tmae)
            print(len(testrmse))



        print(len(testrmse))
        logger.info("Test Set")
        logger.info("RMSE, MAE")
        if refidx in [3,4,8,14]:
            logger.info('{:8.4f} {:8.4f}\n'.format(np.sqrt(np.mean(testrmse)), np.mean(testmae)))
        else:
            logger.info('{:8.4f} {:8.4f}\n'.format(np.sqrt(np.mean(testrmse))*1./(kcal/mol), np.mean(testmae)*1./(kcal/mol)))
def get_mustd(index,refidx):
    for line in open("../prepare_data/db/split_" + str(index) + "/reference.csv"):
        if int(line.split(",")[0]) == refidx:
            mu = np.array([float(line.split(",")[1])])
            std = np.array([float(line.split(",")[2].strip("\n"))])
    return mu, std

def get_ref(refidx):
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
    reference ="../prepare_data/atomrefs.txt.npz"
    reference = np.load(reference)
    atom_ref = reference['atom_ref'][:, ref:(ref+1)]
    return atom_ref


index = int(sys.argv[1])
refidx = int(sys.argv[2])
train_position = str(sys.argv[3])
test_type = str(sys.argv[4])
mu,std = get_mustd(index,refidx)
atom_ref = get_ref(refidx)

evalmodel(index, refidx, train_position, test_type, mu, std, atom_ref)

