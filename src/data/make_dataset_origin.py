# -*- coding: utf-8 -*-
import click
import logging
import gdb9
from gdb9 import load_atomrefs
from gdb9 import load_data
import split_db
from split_db import split_ase_db
import Convert_db_mmff
from Convert_db_mmff import convertXYZ2SDF
import Gen_tfrecord
from Gen_tfrecord import predictLayer
#from pathlib import Path
#from dotenv import find_dotenv, load_dotenv


@click.command()
@click.option("--split_times")
@click.option("--train_size")
@click.option("--val_size")
@click.option("--test_live_size")
@click.option('--input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath, split_times, train_size = 99000, val_size = 1000, test_live_size = 1000):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Prepare raw data")
    datadir_raw = os.path.join(output_filepath,"raw")
    ### Download atom reference and gdb9 data ###
    load_atomrefs(os.path.join(datadir_raw, "atomrefs.txt"))
    load_data(os.path.join(datadir_raw,"gdb9.db"))
    ### Split data ###
    datadir_pro = os.path.join(output_filepath,"process")
    asedb = os.path.join(datadir_raw,"gdb9.db")
    ### Split n times ###
    for i in range(split_times):
        logger.info("Split Data: split_" + str(i+1) + " start")
        dstdir = datadir_pro + "/split_" + str(i+1)
        partitions = {"train":train_size,"validation":val_size,"test_live":test_live_size,"test":-1}
        split_ase_db(asedb,dstdir,partitions)
        logger.info("split_" + str(i+1) + " finish")
    logger.info("Done. ")

    logger.info("Generate MMFF optimized geometries")
    ### Generate MMFF optimized geomerties ###
    for i in range(split_times):
        for t in ["train","val","test_live","test"]:
            dbpath = datadir_pro + "/split_" + str(i+1)
            indb = os.path.join(dbpath, t + ".db")
            outpath = os.path.join(dbpath, t)
            convertXYZ2SDF(indb,outpath)
    logger.info("Done.")

    logger.info("Prepare tfrecord file")
    for i in range(split_times):
        for t in ["train","val","test_live","test"]:
            inpath = datadir_pro + "/split_" + str(i+1) + "/" + t
            outpath = os.path.join(output_filepath,"ready/split_" + str(i+1) + "/")
            outfile = os.path.join(outpath,t + ".tfrecord")
            size = 
            predictLayer(inpath, outfile, size)
    
    logger.info('making final data set from raw data')
    


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
