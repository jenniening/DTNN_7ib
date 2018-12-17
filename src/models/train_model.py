### script used to train model ###
import tensorflow as tf
import numpy as np

import train
from train import read_mustd
from train import get_ref
from train import runmodel

import os
import click
import logging


@click.command()
@click.option("--refidx",default = "10", show_default = True, help = "Target index")
@click.option("--datatype", default = "qm9mmff", show_default = True, help = "Please identify input datatype: qm9mmff,emol9mmff")
@click.option("--geometry", default = "QM", show_default = True, help = "Please identify input geometry: QM, MMFF")
@click.option("--batchsize",default = "100", show_default = True, help = "Training batchsize")
@click.option("--opt",default = "Adam", show_default = True, help = "Training optimizer: Adam, GD")
@click.option("--trainschedule",default = "cyclic", show_default = True, help = "Training schedule: cylcic, constant, decay")
@click.option("--cycles",default = "8", show_default = True, help = "Number of cycles in cyclic training schedule")
@click.option("--decayrate",default = "0.95", show_default = True, help = "Decay rate in decay training schedule")
@click.option("--lrint",default = "0.001", show_default = True, help = "Initial learning rate")
@click.option("--nbasis",default = "256", show_default = True, help = "Number of basis functions")
@click.option("--ninteractions",default = "7", show_default = True, help = "Depth of interaction blocks")
@click.option("--cutoff",default = "3", show_default = True, help = "Distance cutoff")
@click.option("--step",default = "0.1", show_default = True, help = "Step between Gaussians")
@click.option("--activation",default = "SSoftp", show_default = True, help = "Activation function in interaction block")
@click.option("--weights",default = "update", show_default = True, help = "Share or update weights in interaction block")
@click.option("--nepoch",default = "400", show_default = True, help = "Training epochs")
@click.option("--checkpointinterval",default = "1000", show_default = True, help = "Check point every n steps")
@click.option("--inputdir", default = "../../data/external",show_default = True, help = "Input directory", type = click.Path(exists = True))
@click.option("--outputdir",default = "../../models/",show_default = True, help = "Output directory", type = click.Path())
### Whether to add new molecules dataset ###
@click.option("--addnewm",is_flag=True, help = "Add new molecules dataset flag ")
### when transfer learning model (tlmmff,tlconfmmff), use transferlearning flag and provide restorename ###
@click.option("--transferlearning",is_flag=True, help = "Transfer learning flag")
@click.option("--restorename",default = "model_10_qm9mmff_QM_100_Adam_Cyclic_8_0.95_0.001_256_7_3_0.1_SSoftp_update_400", show_default = True, help = "Restore model in transfer learning")
@click.option("--restoredir",default = "../../models/", show_default = True, help = "Restore model directory")



def main(inputdir, outputdir, refidx, datatype, geometry, batchsize, opt, trainschedule, cycles, decayrate, lrint, nbasis, ninteractions, cutoff, step, activation, weights, nepoch, addnewm, transferlearning, restorename, restoredir, checkpointinterval):

    inputdir = os.path.realpath(inputdir)
    inputdir = os.path.join(inputdir,datatype)
    outputdir = os.path.realpath(outputdir)
    restoredir = os.path.realpath(restoredir)
    if datatype == "qm9mmff" and geometry == "QM":
        modeltype = "dtnn7id"
    elif datatype == "qm9mmff" and geometry == "MMFF" and transferlearning:
        modeltype = "tlmmff"
    elif datatype == "emol9mmff" and geometry == "MMFF" and transferlearning:
        modeltype = "tlconfmmff"
    else:
        modeltype = "others"
    outputdir = os.path.join(outputdir,modeltype)
    os.system("mkdir " + outputdir)
    os.chdir(outputdir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("HYPERPARAMETERS:")
    logger.info("Target Index:" + refidx )
    logger.info("Data Type:" + datatype)
    logger.info("Input Geometry:")
    logger.info("Batch Size:" + batchsize )
    logger.info("Optimizer Type:" + opt )
    logger.info("Train Schedule:" + trainschedule)
    if trainschedule == "cyclic":
        logger.info("Cycles:" + cycles)
    if trainschedule == "decay":
        logger.info("Decay Rate" + decayrate)
    logger.info("Initial Learning Rate:" + lrint )
    #logger.info("Initializer Type:" + initialize )
    #logger.info("Dropout:" + str(dropout) )
    #logger.info("Batchnorm:" + str(batchnorm) )
    logger.info("N_basis:" + nbasis )
    logger.info("N_interactions:" + ninteractions)
    logger.info("Distance Cutoff:" + cutoff )
    logger.info("Distance Step:" + step )
    logger.info("Activation Function:" + activation )
    logger.info("Weights:" + weights)
    logger.info("N_epochs:" + nepoch )
    logger.info("Add New Molecules Dataset:" + str(addnewm))
    logger.info("Transfer Learning:" + str(transferlearning))
    if transferlearning:
        logger.info("Restore Model:" + restorename)
        if restorename.split("_")[2] == "qm9mmff" and restorename.split("_")[3] == "QM":
            restore_modeltype = "dtnn7id"
        elif restorename.split("_")[2] == "qm9mmff" and restorename.split("_")[3] == "MMFF":
            restore_modeltype = "tlmmff"
        elif restorename.split("_")[2] == "emol9mmff" and restorename.split("_")[3] == "MMFF":
            restore_modeltype = "tlconfmmff"
        else:
            restore_modeltype = "others"
        restoredir = os.path.join(restoredir,restore_modeltype)
        restorename = os.path.join(restoredir,restorename)

    logger.info("Checkpoint Interval:"+ checkpointinterval)

    tf.reset_default_graph()
    if addnewm:
        train_data = [os.path.join(inputdir,"train.tfrecord"),os.path.join(inputdir,"new_molecules.tfrecord")]
        #train_data = os.path.join(inputdir,"train_new.tfrecord")
        train_size = train_size = sum(1 for example in tf.python_io.tf_record_iterator(train_data[0])) + sum(1 for example in tf.python_io.tf_record_iterator(train_data[1]))
        #train_size = sum(1 for example in tf.python_io.tf_record_iterator(train_data))

    else:
        train_data = os.path.join(inputdir,"train.tfrecord")
        train_size = sum(1 for example in tf.python_io.tf_record_iterator(train_data))

    val_data = os.path.join(inputdir,"validation.tfrecord")
    test_data = os.path.join(inputdir,"testlive.tfrecord")
    val_size = sum(1 for example in tf.python_io.tf_record_iterator(val_data))
    test_size = sum(1 for example in tf.python_io.tf_record_iterator(test_data))

    logger.info("Total train:" + str(train_size) )
    logger.info("Total validation:" + str(val_size) )
    logger.info("Total test:" + str(test_size) )
    if datatype == "emol9mmff":
        inref1 = os.path.join(inputdir,"atomref.B3LYP_631Gd.npz")
    else:
        inref1 = os.path.join(inputdir,"atomrefs.txt.npz")
    
    atom_ref = get_ref(inref1, refidx)
    inref2 = os.path.join(inputdir,"mu_std.csv")
    mu, std = read_mustd(inref2,refidx)
    logger.info("mu:" + str(mu))
    logger.info("std:" + str(std) )
    n_iterations = int(np.ceil(int(nepoch) * train_size/ int(batchsize)))
    logger.info("Total iterations:" + str(n_iterations) )

    ### model name ###
    if transferlearning:
        args = [refidx, datatype, geometry, batchsize, opt, trainschedule, cycles, decayrate, lrint, nbasis, ninteractions, cutoff, step, activation, weights, nepoch, "transfer"]
    else:
        args = [refidx, datatype, geometry, batchsize, opt, trainschedule, cycles, decayrate, lrint, nbasis, ninteractions, cutoff, step, activation, weights, nepoch]
    
    model_name = "model_" + "_".join(args)
    logger.info("MODEL NAME:")
    logger.info(model_name)

    # train position
    train_position_dic = {"qm9mmff_QM":"positions","qm9mmff_MMFF":"mmffpositions","emol9mmff_QM":"positions1","emol9mmff_MMFF":"positions2"}
    train_position = train_position_dic[datatype + "_" + geometry]
    # transfer 


    runmodel(int(refidx), train_position, train_data, val_data, test_data, train_size, val_size, test_size, 
            int(batchsize), int(nbasis), int(ninteractions), float(cutoff), float(step), 
            activation, weights, opt, trainschedule, int(cycles), float(lrint), float(decayrate),int(n_iterations), 
            mu, std, atom_ref, 
            model_name, transferlearning, restorename, logger, int(checkpointinterval),outputdir)
##
if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s -  %(levelname)s - %(message)s'
    datefmt='%m/%d/%Y %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=datefmt)
    main()

