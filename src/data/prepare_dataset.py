### This is scipt to prepare datasets used in our project ###
#  -*- coding: utf-8 -*-
import Prepare_QM9_MMFF
from Prepare_QM9_MMFF import split_data as split_data_qm9
from Prepare_QM9_MMFF import write_tfrecord

import ConvertTFrecord_Platinum_MMFF
from ConvertTFrecord_Platinum_MMFF import processQMSDF as processPlatinum

import ConvertTFrecord_eMol9_MMFF
from ConvertTFrecord_eMol9_MMFF import split_data as split_data_eMol
from ConvertTFrecord_eMol9_MMFF import processQMSDF as processeMol

import Get_mustd
from Get_mustd import getmustd_final

import os
import tarfile
import numpy as np
import click
import logging

#from dotenv import find_dotenv, load_dotenv


@click.command()
@click.option("--datatype", default = "qm9mmff", show_default = True, help = "Please identify which dataset you want to proces:qm9mmff,emol9mmff,platinummmff")
@click.option("--inputdir", default = "../../data/raw",show_default = True, help = "Input directory", type = click.Path(exists = True))
@click.option("--outputdir",default = "../../data/processed",show_default = True, help = "Input directory", type = click.Path())


def main(datatype, outputdir, inputdir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    outputdir = os.path.realpath(outputdir)
    inputdir = os.path.realpath(inputdir)
    datadir = os.path.join(outputdir,datatype)
    os.system("mkdir " + datadir)
    if datatype == "qm9mmff":
        logger.info('QM9_MMFF')
        os.chdir(datadir)
        ### split data ###
        partition = {"train":99000,"validation":1000,"test_live":1000}
        total_num = 133885
        if not os.path.isfile("split_infor.npz"):
            train_id,val_id,test_live_id,test_id = split_data_qm9(partition, total_num, False)
        else:
            split = np.load("split_infor.npz")
            train_id,val_id,test_live_id,test_id = split["train"],split["validation"],split["test_live"],split["test"]
        tar = tarfile.open(os.path.join(inputdir,"qm9_mmff.tar.bz2"))
        ### training ###
        outfile = "train.tfrecord"
        write_tfrecord(tar,outfile,train_id[0:5])
        ### validation ###
        outfile = "validation.tfrecord"
        write_tfrecord(tar,outfile,val_id[0:5])
        ### test_live ###
        outfile = "testlive.tfrecord"
        write_tfrecord(tar,outfile,test_live_id[0:5])
        ### test ###
        outfile = "test.tfrecord"
        write_tfrecord(tar,outfile,test_id[0:5])
        tar.close()
        ### get mu, std ###
        train = "train.tfrecord"
        size = len(train_id[0:5])
        position = "positions"
        reference = os.path.join(inputdir,"atomrefs.txt.npz")
        outfile = open("mu_std.csv","w")
        getmustd_final(train,position,size,reference,outfile)
        logger.info('Done')

    elif datatype == "emol9mmff":
        logger.info("eMol9_MMFF")
        os.chdir(datadir)
        ### split_data ###
        total_file = os.path.join(inputdir,"eMol9_mmff_index.csv")
        partition = {"test":1348,"validation":500,"test_live":500}
        if not os.path.isfile("split_infor.npz"):
            traindf, valdf, testdf, testlivedf = split_data_eMol(total_file, partition["test"], partition["validation"], partition["test_live"])
            traindf,valdf,testdf,testlivedf = traindf["index"].values,valdf["index"].values,testdf["index"].values,testlivedf["index"].values
        else:
            data = np.load("split_infor.npz")
            traindf,valdf,testdf,testlivedf = data["train_idx"], data["validation_idx"],data["test_idx"],data["test_live_idx"]
        ## write set ###
        tar = tarfile.open((os.path.join(inputdir,"eMol9_MMFF.tar.bz2")))
        ### training ###
        outtfile = 'eMol9_mmff_train.tfrecord'
        processeMol(tar,outtfile,traindf[0:5])
        ### validation ###
        outtfile = 'eMol9_mmff_validation.tfrecord'
        processeMol(tar,outtfile,valdf[0:5])
        ### test_live ###
        outtfile = 'eMol9_mmff_testlive.tfrecord'
        processeMol(tar,outtfile,testlivedf[0:5])
        ### test ###
        outtfile = 'eMol9_mmff_test.tfrecord'
        processeMol(tar,outtfile,testdf[0:5])
        tar.close()
        logger.info('Done')
        ### get mu,std ###
        train = "eMol9_mmff_train.tfrecord"
        size = len(traindf[0:5])
        position = "positions1"
        reference = os.path.join(inputdir,"atomref.B3LYP_631Gd.npz")
        outfile = open("mu_std.csv","w")
        getmustd_final(train,position,size,reference,outfile)
        logger.info('Done')


    elif datatype == "platinummmff":
        logger.info("Platinum_MMFF")
        ### no split for PlatinumMMFF dataset ###
        ### all structures have been used as test ###
        os.chdir(datadir)
        tar = tarfile.open(os.path.join(inputdir,"Platinum_MMFF.tar.bz2"))
        rmsd = os.path.join(inputdir,"RMSD_nolarger01.csv")
        index_list = [line.split(",")[1].rstrip() for line in open(rmsd) if line.split(",")[1].rstrip() != "index"][0:5]
        pro = os.path.join(inputdir,"Gaussian_properties.csv")
        outtf = "platinum.tfrecord"
        processPlatinum(tar,pro,outtf,index_list)
        tar.close()
        logger.info('Done')




    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s -  %(levelname)s - %(message)s'
    datefmt='%m/%d/%Y %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=datefmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    main()
