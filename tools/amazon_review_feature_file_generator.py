#!/bin/python
# TODO: remove this file - defunct and replaced with feature_generator.py
#
# Generate feature files based on a config
#
import sys
sys.path.append('../')

import argparse
import pandas as pd
import logging
import traceback2
from pprint import pformat
import numpy as np
from datetime import datetime

LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s:%(lineno)d %(levelname)s - %(message)s'

# KEEP_COLUMNS = ["product_title", "helpful_votes", "review_headline", "review_body", "star_rating"]
# FEATURE_COLUMNS = ["review_headline", "review_body"]
# FEATURE_COLUMNS = ["review_body"]

# csv that has all the input
# PARAM_INFILE = "amazon_review_feature_generation_input.csv"
TRUE_NAMES = ["yes", "Yes", "True", "true"]


# TODO: forgot to add back in the classlification to the feature file (Y)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes pre-processed files and generate feature files')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    # parser.add_argument("infile", help="data file to generate features from")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    parser.add_argument("-c", "--column", help="column to generate features from", default="review_body")
    parser.add_argument("-o", "--outdir", help="output director", default="dataset/feature_files")
    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)

    start_time = datetime.now()
    config_df = pd.read_csv(args.config_file, dtype={"min_df": np.float16,
                                                     "max_df": np.float16})
    config_df = config_df[["description",
                           "fn_name",
                           "lda_topics",
                           "min_df",
                           "max_df",
                           "min_ngram_range",
                           "max_ngram_range",
                           "max_features",
                           "feature_columns",
                           "y"]]
    df = pd.read_csv(args.infile)

    for index, row in config_df.iterrows():
        columns = row["feature_columns"].replace(" ", "").split(",")
        log.info(f'feature columns: {columns}')

        y_columns = row["y"].replace(" ", "").split(",")
        log.info(f'y columns: {y_columns}')

        for column in columns:
            x_df = df[column]
            y_df = df[y_columns]

            try:
                # get function name to call
                function = row.fn_name

                # prepare arguments

                # had issues with min_df and max_df since it can be int or float - namely there is
                # functional difference if you pass in 1.0 vs 1
                # instead rely on default values set in the functions and drop the columns if they are
                # not defined
                args_pd = row.dropna()
                args_dict = args_pd.to_dict()
                # delete columns from config file that doesn't get passed to feature fn
                del args_dict["fn_name"]
                del args_dict["feature_columns"]
                del args_dict["y"]

                args_dict["feature_column"] = column
                log.info(f'generating features from source file: {args.infile}')
                log.info(f'generating file with following arguments: {pformat(args_dict)}')
                args_dict["x"] = x_df
                args_dict["y"] = y_df

                # call function specified by the fn_name column
                _, v_time, vfile_time, lda_time, ldaf_time = globals()[function](**args_dict)
                config_df.loc[index, "vectorize_time_min"] = v_time
                config_df.loc[index, "vectorize_file_time_min"] = vfile_time
                config_df.loc[index, "lda_time_min"] = lda_time
                config_df.loc[index, "lda_file_time_min"] = ldaf_time
                config_df.loc[index, "status"] = "success"
            except Exception as e:
                log.error(traceback2.format_exc())
                config_df.loc[index, "status"] = "failed"
                config_df.loc[index, "message"] = str(e)
            finally:
                config_df.to_csv(args.config_file, index=False)


    end_time = datetime.now()
    total_time_min = end_time - start_time
    print(f'Finished processing. Total time: {round(total_time_min.total_seconds() / 60, 1)} min')
