"""
Use this program to zip report files together.

Reason for this is things change over time, the report columns are changing
it is easier to use pandas to zip reports to match columns than to do this manually

We will be taking one file and adding the entries from another file into this

"""
import sys
sys.path.append('../')


import pandas as pd
import logging
import argparse

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TRUE_LIST = ["yes", "Yes", "YES", "y", "True", "true", "TRUE"]
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add a class column ')
    parser.add_argument("outfile", help="other file to draw data from")
    # TODO: make this handle any number of files
    parser.add_argument("file1", help="file to zip into")
    parser.add_argument("file2", help="other file to draw data from")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)

    log.info(f'Reading {args.file1}')
    in_pd = pd.read_csv(args.file1)
    log.info(f'Reading {args.file2}')
    from_pd = pd.read_csv(args.file2)

    new_df = in_pd.append(from_pd, ignore_index=True)
    log.info(f'Resulting shape {new_df.shape}')
    log.info(f'Writing to {args.outfile}')
    new_df.to_csv(args.outfile, index=False)
