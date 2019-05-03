#
#

import os
import argparse
import pandas as pd
import logging
from TextPreprocessor import TextPreprocessor
import re


# set up logging
LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
    " [%(lineno)d] - %(message)s"
logger = logging.getLogger(__name__)

STOP_WORDS_TO_REMOVE=[
    'no',
    'not',
    'do',
    'does',
    'did',
    'does',
    'should',
    'very'
    'will'
    ]

def remove_amazon_tags(text: str) -> str:
    """
    removes amazon tags that look like [[VIDEOID:dsfjljs]], [[ASIN:sdjfls]], etc
    :param text:
    :return:
    """
    logger.debug(f"before amazon tags {text}")
    text = re.sub('\[\[.*?\]\]', ' ', text, re.I | re.A)
    text = ' '.join(text.split())
    logger.debug(f"after processing amazon tags {text}")
    return text


def main():
    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="source data file")
    parser.add_argument("-o", "--outfile", help="output file", default="outfile.csv")
    parser.add_argument("-l", "--loglevel", help="log level")
    parser.add_argument("-r", "--retain", help="log level", action="store_true", default=False)
    parser.add_argument("-c", "--convert", action='store_true',
                    help="convert to csv")
    # get command line arguments
    args = parser.parse_args()

    # process argument
    loglevel = logging.INFO
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    logger = logging.getLogger(__name__)

    infile = args.datafile
    outfile = args.outfile
    assert os.path.isfile(infile), f"{infile} does not exist"

    logger.info(f'loading data frame from {infile}')
    df = pd.read_csv(infile, ",", parse_dates=["review_date"])
    logger.info(f'finished loading dataframe {infile}')
    logger.info(f'original dataframe length: {len(df)}')

    tp = TextPreprocessor(text_columns=["product_title", "review_headline", "review_body"],
                          columns_to_drop=['marketplace', 'vine', 'verified_purchase'],
                          stop_word_remove_list=STOP_WORDS_TO_REMOVE,
                          create_original_columns=args.retain,
                          custom_preprocessor=remove_amazon_tags)
    df = tp.preprocess_data(df)
    logger.info(f'new dataframe length: {len(df)}')

    logger.info(f'writing dataframe to {outfile}')
    df.to_csv(outfile, doublequote=True, index=False)


if __name__ == '__main__':
    main()
