#!/bin/python
#
# Generate feature files based on a config
#
import pandas as pd
# leave this in - will be called by globals
from nlp.feature_util import generate_bow_file, generate_tfidf_file, generate_word2vec_file, generate_fasttext_file
import logging
from pprint import pformat
from util.ConfigBasedProgram import ConfigBasedProgram, TimedProgram

log = logging.getLogger(__name__)
OUTFILE = "outfile"


class GenerateWord2Vec(object):

    def __init__(self, config_df: pd.DataFrame):
        super.__init__(self)
        self.config_df = config_df


class GenerateFeatures(TimedProgram):

    def execute(self):
        log.info("Execute")

        feature_columns = self.get_config_list("feature_columns")
        function = self.get_config("fn_name")
        y_columns = self.get_config_list("y")

        infile = self.get_infile()

        df = pd.read_csv(infile)

        for feature_column in feature_columns:
            # get x column
            x_df = df[feature_column]
            y_df = df[y_columns]

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

            args_dict["feature_column"] = feature_column
            log.info(f'generating features from source file: {infile}')
            log.info(f'generating file with following arguments: {pformat(args_dict)}')
            args_dict["x"] = x_df
            args_dict["y"] = y_df
            args_dict["timer"] = self.report

            # call function specified by the fn_name column
            outfile = globals()[function](**args_dict)
            self.record(OUTFILE, outfile)


if __name__ == "__main__":
    prog = ConfigBasedProgram("Takes pre-processed files and generate feature files", GenerateFeatures)
    prog.add_argument("-o", "--outdir", help="output director", default="dataset/feature_files")
    prog.main()
