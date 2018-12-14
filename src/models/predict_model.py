### script used to test model ###
import tensorflow as tf
import numpy as np

import test
from test import evalmodel

import os
import click
import logging


@click.command()
@click.option("--refidx",default = "10", show_default = True, help = "Predict index")
@click.option("--batchsize",default = "100", show_default = True, help = "Batchsize")
@click.option("--modelname", default = "",show_default = True, help = "Model name")
@click.option("--modeldir",defult = "../model", show_default = True, help = "Model directory" )
@click.option("--testtype", default = "qm9mmff", show_default = True, help = "Test type:qm9mmff,emol9mmff,platinummmff")
@click.option("--testpositions", default = "positions", show_default = True, help = "Input positions: qm9mmff-->(positions or mmffpositions, emol9mmff/platinummmff-->(positions1 or positions2)")
@click.option("--modelselect", default = "val", show_default = True, help = "Value used to chose final model: val, total")
@click.option("--inputdir", default = "../../data/processed",show_default = True, help = "Input directory", type = click.Path(exists = True))
@click.option("--outputdir",default = "../../models/",show_default = True, help = "Output directory", type = click.Path())

def main(inputdir, outputdir, refidx, batchsize, modelname, testtype, testpositions, modelselect, transferlearning, restorename):
    inputdir = os.path.realpath(inputdir)
    inputdir = os.path.join(inputdir,testtype)
    args = modelname.split("_")[1:]
    if "transfer" in args:
        transferlearning = True
        restorename = "_".join(args[args.index("transfer") + 1:])
    if args[0] == "qm9mmff" and args[1]== "QM":
        modeltype = " dtnn7id"
    elif args[0] == "qm9mmff" and args[1] == "MMFF" and transferlearning:
        modeltype = "tlmmff"
    elif args[0] == "emol9mmff" and args[1] == "MMFF" and transferlearning:
        modeltype = "tlconfmmff"
    else:
        modeltype = "others"
        

    
    outputdir = os.path.realpath(outputdir)



    evalmodel(refidx, test_position, test_data, test_size, test_type,
              batch_size,model_name, model_dir,
              transfer_learning, mu, std, atom_ref,logger)
