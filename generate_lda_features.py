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
from util.df_util import cast_samller_type
import sys


LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s:%(lineno)d %(levelname)s - %(message)s'
log = None

FEATURE_COLUMN="review_body"
CLASS_COLUMN="star_rating"


# In[2]:


# around 100k entries
CONFIG_FILE = "2019-05-21-amazon_review_generate_lda_feature_input.csv"
INFILES = [ "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-10.csv"]
TOPICS = [ 40, 60 ]
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
    parser.add_argument("-t", "--traindebug", help="run data through training models for debugging", action="store_true")

    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)

    # read in configuration
    config = pd.read_csv(args.config_file)
    log.debug(config.head())
    for index, row in config.iterrows():
        infile = row["infile"]
        dtype = row["dtype"]

        # infile has a bunch of feature columns, last column is star_rating
        log.info(f'Reading in {infile}')
        in_df = pd.read_csv(infile)

        log.info(f'Casting df to smaller type...')
        if dtype or len(dtype) > 0:
            in_df = cast_samller_type(in_df, CLASS_COLUMN, dtype)

        log.info(f'Cleaning data...')
        Y = in_df[CLASS_COLUMN]
        X = in_df.drop(CLASS_COLUMN, axis=1)

        for topics in TOPICS:
            features = generate_lda_feature(X, Y, topics)
            log.info(f"Feature shape: {features.shape}")
            outfile = f'{infile.split(".")[0]}-lda{topics}.csv'
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

            zip_and_save(features, Y, outfile)





