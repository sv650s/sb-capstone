#!/usr/bin/env python
# coding: utf-8

# # Prototype to use topic modeling as a means for feature generation
# 
# 
# Goal:
#     * prototype topic modeling (LDA) as a means for feature generation
#     * run feature matrix through some classification models and evaluate results
# 
# 
# References:
# 
# * [Traditional Methods for Text Data](https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41)
# * [An overview of topics extraction in Python with LDA](https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc)
# * [gensim](https://radimrehurek.com/gensim/)
# * [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/)


#
# configuration file format
# infile - input for this program is a word vector file generated from bow or tfidf
# outfile - file to write the feature output to
#

# In[1]:


from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import argparse
import re
from pprint import pprint
import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from util.dict_util import add_dict_to_dict
import util.df_util as dfu
import numpy as np
import sys
import traceback2
from datetime import datetime


LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s:%(lineno)d %(levelname)s - %(message)s'
log = None

FEATURE_COLUMN="review_body"
CLASS_COLUMN="star_rating"


# In[2]:


# around 100k entries
# CONFIG_FILE = "2019-05-21-amazon_review_generate_lda_feature_input.csv"
# INFILES = [ "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-10.csv"]
# TOPICS = [ 20, 40, 60 ]
#TOPICS = [ 5, 10, 20 ]



def get_outfile_name(infile:str) -> str:
    # generate output file name
    # dir = re.findall(r'([\w/]+)/[\w-]+\.csv', INFILE)[0]
    # print(f'dir: {dir}')
    basename = re.findall(r'/([\w-]+)\.csv', file)[0]
    log.debug(f'basename: {basename}')
    basename = basename.replace('preprocessed', f'lda{topic}')
    log.debug(f'basename: {basename}')
    outfile = f'dataset/feature_files/{basename}.csv'
    log.debug(f'outfile: {outfile}')
    return outfile


# TODO: add this to Amazon preprocessor
def clean_mixed_words(x):
    # remove mixed words
    x = re.sub(r'\s+([a-z]+[\d]+[\w]*|[\d]+[a-z]+[\w]*)', '', x)
    # remove numbers
    x = re.sub(r'\s(\d+)', '', x)
    return x




def generate_lda_feature(x, y, topic) -> pd.DataFrame:
    log.info(f"Generating lda features from x:{x.shape} y:{y.shape} topics:{topic}")
    lda = LatentDirichletAllocation(n_components=topic, max_iter=10, random_state=0)
    dt_matrix = lda.fit_transform(x)
    return pd.DataFrame(dt_matrix)

def zip_and_save(x, y, outfile):
    log.info("Zipping features and classes")
    x.loc[:, CLASS_COLUMN] = y
    log.info(f"Saving file: {outfile}")
    x.to_csv(outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes word vector files (ie, BoW and TFIDF and generate feature files based on LDA')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    parser.add_argument("-o", "--outdir", help="output directory", default="dataset/feature_files")
    parser.add_argument("-t", "--traindebug", help="run data through training models for debugging", action="store_true")

    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)

    # read in configuration
    config_df = pd.read_csv(args.config_file)
    log.debug(config_df.head())
    for index, row in config_df.iterrows():
        infile = row["infile"]
        data_dir = row["data_dir"]
        dtype = row["dtype"]
        topics_str_list = row["topics"].replace(" ", "").split(",")
        topics = list(map(int, topics_str_list))


        read_infile_start = None
        read_infile_end = None
        lda_start = None
        lda_end = None
        save_start = None
        save_end = None

        try:

            # time reading in file time
            read_infile_start = datetime.now()

            # infile has a bunch of feature columns, last column is star_rating
            log.info(f'Reading in {infile}')
            if dtype and len(dtype) > 0:
                in_df = pd.read_csv(f'{data_dir}/{infile}', dtype=np.dtype(dtype))
                in_df = dfu.cast_column_type(in_df, CLASS_COLUMN, "uint8")
            else:
                in_df = pd.read_csv(f'{data_dir}/{infile}')

            read_infile_end = datetime.now()

            log.info(f'Cleaning data...')
            Y = in_df[CLASS_COLUMN]
            X = in_df.drop(CLASS_COLUMN, axis=1)

            for topic in topics:
                lda = f"lda{topic}"

                try:

                    # time lda start
                    lda_start = datetime.now()
                    features = generate_lda_feature(X, Y, topic)
                    lda_end = datetime.now()

                    log.info(f"Feature shape: {features.shape}")
                    outfile = f'{args.outdir}/{infile.split(".")[0]}-{lda}.csv'
                    log.info(f"outfile: {outfile}")

                    # do remove this later
                    if args.traindebug:
                        X_train, X_test, y_train, y_test = train_test_split(features, Y)

                        log.info("Training KNN model")
                        neigh = KNeighborsClassifier(n_jobs=-1)
                        neigh.fit(X_train, y_train)
                        y_predict = neigh.predict(X_test)

                        report = {}
                        log.info("Getting classification report")
                        lda_dict = classification_report(y_test, y_predict, output_dict=True)
                        report = add_dict_to_dict(report, lda_dict)
                        print(type(report))
                        pprint(report)
                        report_df = pd.DataFrame(report, index=[0])
                        report_df.to_csv('lda_knn_report.csv', index=False)

                    # time save time
                    save_start = datetime.now()
                    zip_and_save(features, Y, outfile)
                    save_end = datetime.now()
                    config_df.loc[index, "status"] = "success"

                finally:
                    if save_start and save_end:
                        config_df.loc[index, f"save_time_{lda}"] = \
                            round((save_end - save_start).total_seconds() / 60, 2)
                    elif save_start:
                        config_df.loc[index, f"save_time_{lda}"] = \
                            round((datetime.now() - save_start).total_seconds() / 60, 2)

                    if lda_start and lda_end:
                        config_df.loc[index, f"lda_time_{lda}"] = \
                            round((lda_end - lda_start).total_seconds() / 60, 2)
                    elif lda_start:
                        config_df.loc[index, f"lda_time_{lda}"] = \
                            round((datetime.now() - lda_start).total_seconds() / 60, 2)

        except Exception as e:
            traceback2.print_exc(file=sys.stderr)
            config_df.loc[index, "status"] = "failed"
            config_df.loc[index, "message"] = str(e)
        finally:
            if read_infile_start and read_infile_end:
                config_df.loc[index, "read_time"] = \
                        round((read_infile_end - read_infile_start).total_seconds() / 60, 2)
            elif read_infile_start:
                config_df.loc[index, "read_time"] = \
                        round((datetime.now() - read_infile_start).total_seconds() / 60, 2)

            # write status of each round
            config_df.to_csv(args.config_file, index=False)




