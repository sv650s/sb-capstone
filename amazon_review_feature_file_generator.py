#!/bin/python
#
# Generate feature files based on a config
#
import argparse
import pandas as pd
from nlp.feature_util import generate_bow_file, generate_tfidf_file
import logging
import traceback2
from pprint import pformat
import numpy as np
import datetime

LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s:%(lineno)d %(levelname)s - %(message)s'

# INFILES = [
#     "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-supertiny.csv",
#     "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-tiny.csv",
#     "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-small.csv",
#     "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-medium.csv",
# ]
# INFILES = [
#     "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-tiny.csv",
#     "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-small.csv",
#     ]
# for debugging
# INFILES = [
#      "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-supertiny.csv"
# ]
KEEP_COLUMNS = ["product_title", "helpful_votes", "review_headline", "review_body", "star_rating"]
OUTDIR = "dataset/feature_files"
# FEATURE_COLUMNS = ["review_headline", "review_body"]
FEATURE_COLUMNS = ["review_body"]

# csv that has all the input
# PARAM_INFILE = "amazon_review_feature_generation_input.csv"

def read_amazon_data(file:str) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["review_date"])
    return df[KEEP_COLUMNS]


# TODO: forgot to add back in the classlification to the feature file (Y)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes pre-processed files and generate feature files')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("infile", help="data file to generate features from")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    parser.add_argument("-c", "--column", help="column to generate features from", default="review_body")
    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)

    start_time = datetime.now()
    param_df = pd.read_csv(args.config_file, dtype={"min_df":np.float16, "max_df":np.float16})

    for column in FEATURE_COLUMNS:
        df = pd.read_csv(args.infile, parse_dates=["review_date"])
        x_df = df[column]
        y_df = df["star_rating"]

        for index, row in param_df.iterrows():
            try:
                function = row.fn_name

                # prepare arguments

                # had issues with min_df and max_df since it can be int or float - namely there is
                # functional difference if you pass in 1.0 vs 1
                # instead rely on default values set in the functions and drop the columns if they are
                # not defined
                args_pd = row.dropna()
                args_dict = args_pd.to_dict()
                del args_dict["fn_name"]

                args_dict["feature_column"] = column
                log.info(f'generating features from source file: {args.infile}')
                log.info(f'generating file with following arguments: {pformat(args_dict)}')
                args_dict["x"] = x_df
                args_dict["y"] = y_df

                # call function specified by the fn_name column
                globals()[function](**args_dict)
            except Exception as e:
                log.error(traceback2.format_exc())

    end_time = datetime.now()
    total_time_min = end_time - start_time
    print(f'Finished processing. Total time: {round(total_time_min.total_seconds() / 60, 1)} min')
