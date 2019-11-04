'''
This program parses through a text file and create a list of unique contractions
Contractions is defined by words with ' in the middle
'''
import sys
sys.path.append('../')

import logging
import argparse
import os
import sys
import util.text_util as tu
import pandas as pd


# set up logging
LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
    " [%(lineno)d] - %(message)s"
log = logging.getLogger(__name__)




def get_contractions_from_file(infile:str) -> dict:
    """
    returns a dictionary with the contraction as key and number of occurance as value
    :param infile:
    :return:
    """
    con_dict = {}

    with open(infile, 'r') as file:
        for line in file:
            contractions = tu.get_contractions(line)
            for word in contractions:
                if word in con_dict.keys():
                    counter = con_dict[word]
                    counter += 1
                    con_dict[word] = counter
                else:
                    con_dict[word] = 1


    return con_dict


def main():

    # set up parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="input file")
    parser.add_argument("--noheader", help="input file", default=False, action='store_true')
    parser.add_argument("-l", "--loglevel", help="log level", default="INFO")

    args = parser.parse_args()

    # set up logger
    numeric_log_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=numeric_log_level)
    log = logging.getLogger(__name__)


    if not os.path.isfile(args.infile):
        log.error(f'{infile} does not exist')
        sys.exit(1)

    con_dict = get_contractions_from_file(args.infile)
    log.info(con_dict)

    df = pd.DataFrame.from_records([con_dict]).reset_index().transpose()
    log.info(df.info())
    df.to_csv("contractions.csv")




if __name__ == "__main__":
    main()
