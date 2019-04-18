#
#

import os
import getopt
import argparse
import pandas as pd
import logging
import sys
import unittest2
import yelp_util as yu


# set up logging
LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
    " [%(lineno)d] - %(message)s"
logger = None

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("datafile", help="source data file")
parser.add_argument("outfile", help="output file")
parser.add_argument("-l", "--loglevel", help="log level")
parser.add_argument("-c", "--convert", action='store_true',
                    help="convert to csv")

# file to parse
infile = None
outfile = None


def main():
    # get command line arguments
    args = parser.parse_args()

    # process argument
    loglevel = logging.ERROR
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    logger = logging.getLogger(__name__)

    infile = args.datafile
    outfile = args.outfile
    assert os.path.isfile(infile), f"{infile} does not exist"

    logger.debug(f'starting to read {infile}')
    df = pd.read_json(infile, lines=True)
    logger.debug(f'finished reading {infile}')

    if args.convert:
        yu.convert_to_csv(df, outfile)
    else:
        df = yu.process_data(df)
        df.to_csv(outfile, doublequote=True, index=False)



if __name__ == '__main__':
    main()
