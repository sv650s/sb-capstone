import logging
import pandas as pd
import argparse
import numpy as np


TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add a class column ')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)



    