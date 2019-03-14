### script used to test model ###
import tensorflow as tf
import numpy as np

import test
from test import evalmodel
from train import read_mustd
from train import get_ref

import os
import click
import logging


@click.command()
@click.option("--refidx",default = "10", show_default = True, help = "Predict index")
@click.option("--batchsize",default = "100", show_default = True, help = "Batchsize")
@click.option("--modelname", default = "model_10_qm9mmff_QM_100_Adam_Cyclic_8_0.95_0.001_256_7_3_0.1_SSoftp_update_400",show_default = True, help = "Model name")
@click.option("--modeldir",default = "../../models", show_default = True, help = "Model directory" )
@click.option("--testtype", default = "qm9mmff", show_default = True, help = "Test type:qm9mmff,emol9mmff,platinummmff")
@click.option("--testpositions", default = "positions", show_default = True, help = "Input positions: qm9mmff-->(positions or mmffpositions, emol9mmff/platinummmff-->(positions1 or positions2)")
@click.option("--modelselect", default = "val", show_default = True, help = "Value used to chose final model: val, total")
@click.option("--inputdir", default = "../../data/external",show_default = True, help = "Input directory", type = click.Path(exists = True))
@click.option("--outputdir",default = "../../models/",show_default = True, help = "Output directory", type = click.Path())

def main(inputdir, outputdir, refidx, batchsize, modelname, modeldir, testtype, testpositions, modelselect):
    logging.basicConfig(level=logging.INFO) 
    logger = logging.getLogger(__name__)
    
    inputdir = os.path.realpath(inputdir)
    args = modelname.split("_")[1:]
    if "transfer" in args:
        transfer_learning = True
    else:
        transfer_learning = False

    if args[1] == "qm9mmff" and args[2]== "QM":
        modeltype = "dtnn7id"
    elif args[1] == "qm9mmff" and args[2] == "MMFF" and transfer_learning:
        modeltype = "tlmmff"
    elif args[1] == "emol9mmff" and args[2] == "MMFF" and transfer_learning:
        modeltype = "tlconfmmff"
    else:
        modeltype = "others"

    if testtype == "emol9mmff" or testtype == "platinummmff":
        inref1 = os.path.join(os.path.join(inputdir,args[1]),"atomref.B3LYP_631Gd.npz")
    else:
        inref1 = os.path.join(os.path.join(inputdir,args[1]),"atomrefs.txt.npz")
    atom_ref = get_ref(inref1, refidx)
    inref2 = os.path.join(os.path.join(inputdir,args[1]),"mu_std.csv")
    mu, std = read_mustd(inref2,refidx)
    logger.info("mu:" + str(mu))
    logger.info("std:" + str(std) )

    
    inputdir = os.path.join(inputdir,testtype)
    args = modelname.split("_")[1:]
    model_dir =os.path.realpath(os.path.join(modeldir,modeltype))
    outputdir = os.path.realpath(outputdir)
    if testtype == "qm9mmff" or testtype == "emol9mmff":
        test_data = [os.path.join(inputdir,"test.tfrecord"),os.path.join(inputdir,"testlive.tfrecord")]
        test_size1 = sum(1 for exmaple in tf.python_io.tf_record_iterator(test_data[0]))
        test_size2 = sum(1 for exmaple in tf.python_io.tf_record_iterator(test_data[1]))
        test_size = test_size1 + test_size2
    else:
        test_data = [os.path.join(inputdir,"test.tfrecord")]
        test_size1 = sum(1 for exmaple in tf.python_io.tf_record_iterator(test_data[0]))
        test_size = test_size1
    logger.info("Test Data Size:"  + str(test_size))



    evalmodel(refidx, testpositions, test_data, test_size, modelselect,
              batchsize,modelname, model_dir,
              transfer_learning, mu, std, atom_ref,logger)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s -  %(levelname)s - %(message)s'
    datefmt='%m/%d/%Y %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=datefmt)
    main()
