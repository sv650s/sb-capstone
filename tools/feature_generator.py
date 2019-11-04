#!/bin/python
#
# Generate feature files based on a config
#
# See config/template-feature_generator.csv for example
#
# Required columns
#   data_dir - location of pre-processed files
#   data_file - pre-processed files to use
#   fn_name - this will be used to dynamically call imported feature_util functions (see import below)
#   feature_columns - name of column where our features will be generated from
#   y - list of columns that may contain your model labels
#
# Additional columns in the file should match parameter names for the nlp.feature_util function parameters
# The program will parse out these columns and dynamically pass them into the function as parameters
#
# For generate_tfidf_file and generate_bow_file, the following columns are supported
# 	lda_topics	min_df	max_df	min_ngram_range	max_ngram_range	max_features
# For generate_word2vec_file and generate_fasttext_file, the following columns are supported
# 	feature_size	window_context	min_word_count	sample	iterations
#
import sys
sys.path.append('../')

import pandas as pd
# leave this in - will be called by globals
from util.nlp_util import generate_bow_file, generate_tfidf_file, generate_word2vec_file, generate_fasttext_file
import logging
from pprint import pformat
from util.program_util import ConfigFileBasedProgram, TimedProgram

log = logging.getLogger(__name__)
OUTFILE = "outfile"


# class GenerateWord2Vec(object):
#
#     def __init__(self, config_df: pd.DataFrame):
#         super.__init__(self)
#         self.config_df = config_df


class GenerateFeatures(TimedProgram):
    """
    Program to generate feature files from pre-processed file
    """

    def execute(self):
        log.info("Execute")

        feature_columns = self.get_config_list("feature_columns")
        function = self.get_config("fn_name")
        y_columns = self.get_config_list("y")

        infile = self.get_infile()

        df = pd.read_csv(infile)

        for feature_column in feature_columns:
            # get x column
            x_df = df[feature_column].copy()
            y_df = df[y_columns].copy()

            # had issues with min_df and max_df since it can be int or float - namely there is
            # functional difference if you pass in 1.0 vs 1
            # instead rely on default values set in the functions and drop the columns if they are
            # not defined
            args_pd = self.config_df.dropna()
            args_dict = args_pd.to_dict()

            # delete columns from config file that doesn't get passed to feature fn
            del args_dict["fn_name"]
            del args_dict["feature_columns"]
            del args_dict["y"]
            del args_dict["data_dir"]
            del args_dict["data_file"]

            # these start status columns that the program will write - delete them if they are there
            if "status" in args_dict:
                del args_dict["status"]
            if "status_date" in args_dict:
                del args_dict["status_date"]
            if "message" in args_dict:
                del args_dict["message"]

            args_dict["feature_column"] = feature_column
            log.info(f'generating features from source file: {infile}')
            log.info(f'generating file with following arguments: {pformat(args_dict)}')
            args_dict["x"] = x_df
            args_dict["y"] = y_df
            # args_dict["timer"] = self.report

            # call function specified by the fn_name column
            outfile = globals()[function](**args_dict)
            self.record(OUTFILE, outfile)


if __name__ == "__main__":
    prog = ConfigFileBasedProgram("Takes pre-processed files and generate feature files", GenerateFeatures)
    prog.add_argument("-o", "--outdir", help="output director. Default ../dataset/feature_files", default="../dataset/feature_files")
    prog.main()
