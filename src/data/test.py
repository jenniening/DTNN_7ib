import logging
import make_dataset
from make_dataset import test

log_fmt = '%(asctime)s - %(name)s -  %(levelname)s - %(message)s'
datefmt='%m/%d/%Y %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=datefmt)
test()
