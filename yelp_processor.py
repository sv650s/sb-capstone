#
#

import os
import getopt
import argparse
import pandas as pd
import logging
import sys
import unittest2
from TextPreprocessor import TextPreprocessor


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

    logger.debug(f'loading data frame from {infile}')
    df = pd.read_json(infile, lines=True)
    logger.debug(f'finished loading dataframe {infile}')

    if args.convert:
        tp = TextPreprocessor(text_column_name="text")
        tp.convert_to_csv(df, outfile)
    else:
        tp = TextPreprocessor(text_column_name="text",
                              columns_to_drop=['cool', 'date',
                                               'funny', 'useful', 'user_id'],
                              to_lowercase=True,
                              remove_newlines=True,
                              remove_html_tags=True,
                              remove_accented_chars=True,
                              expand_contractions=True,
                              remove_special_chars=True,
                              stem_text=True,
                              remove_stop_words=True)
        df = tp.preprocess_data(df)
        df.to_csv(outfile, doublequote=True, index=False)


if __name__ == '__main__':
    main()
