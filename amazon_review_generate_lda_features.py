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



LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s:%(lineno)d %(levelname)s - %(message)s'
log = None

FEATURE_COLUMN="review_body"
CLASS_COLUMN="star_rating"


# In[2]:


# around 100k entries
CONFIG_FILE = "2019-05-21-amazon_review_generate_lda_feature_input.csv"
INFILES = [ "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-preprocessed-10.csv"]
TOPICS = [ 20 ]
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


# for file in INFILES:

    # outfile = get_outfile_name(file)
    # log.info(f'outfile: {outfile}')
    #
    # df = pd.read_csv(file, parse_dates=["review_date"])
    # review_body_df = df[FEATURE_COLUMN]
    # Y = df[CLASS_COLUMN]
    #
    #
    # review_body_df = review_body_df.apply(clean_mixed_words)
    # review_body_df.head(10)
    #
    #
    # # In[7]:
    #
    #
    # # create BoW features
    # vectorizer = CountVectorizer()
    # feature_matrix = vectorizer.fit_transform(review_body_df)
    # feature_array = feature_matrix.toarray()
    # vocab = vectorizer.get_feature_names()
    #
    # print(f'feature_array shape: {feature_array.shape}')


    # In[8]:
    #
    #
    # print(vocab)
    #
    #
    # # In[9]:
    #
    #
    # dictionary_LDA = dict(enumerate(vocab))
    # print(dictionary_LDA)

    #
    # # In[10]:
    #
    #
    # len(dictionary_LDA.keys())


    # # Couldn't Quite get gensim to work

    # In[11]:

    #
    # print("printing feature array")
    # count = 0
    # for d, doc in enumerate(feature_array):
    #     print(f'd: {d} doc: {doc}')
    #     count += 1
    #     if count > 6:
    #         break


    # In[12]:


    # num_topics = 20
    # lda_model = models.LdaModel(feature_array,
    #                             num_topics=num_topics, \
    # #                                   id2word=dictionary_LDA, \
    #                                   passes=4)


    # # Trying sklearn LatenDirichletAllocation

    # In[13]:
    #
    #
    # for topic in TOPICS:
    #
    #     # generating convert to LDA vector
    #
    #     lda = LatentDirichletAllocation(n_components=topic, max_iter=10, random_state=0)
    #     dt_matrix = lda.fit_transform(feature_matrix)
    #
    #
    #     # In[ ]:
    #
    #
    #     features = pd.DataFrame(dt_matrix)
    #     # features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'])
    #     features.to_csv(outfile, index=False)
    #     # features.to_csv('lda_dt-preprocessed.csv', index=False)
    #

        # In[ ]:


        # look at topics > 0.6
        # tt_matrix = lda.components_
        # for topic_weights in tt_matrix:
        #     topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
        #     topic = sorted(topic, key=lambda x: -x[1])
        #     topic = [item for item in topic if item[1] > 0.6]
        #     print(topic)
        #     print()
        # # rename columns to use words
        # column_mapper = dict(enumerate(vocab))
        #
        # tt_df = pd.DataFrame(tt_matrix)
        #
        # print(f'converated lda feature shape {tt_df.shape}')
        # tt_df.rename(mapper=column_mapper, axis=1, inplace=True)
        # tt_df.to_csv('lda_tt-preprocessed.csv', index=False)



        # # Let's run KNN

        # In[ ]:






        # In[ ]:


def generate_lda_feature(x, y, topic) -> pd.DataFrame:
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

        # infile has a bunch of feature columns, last column is star_rating
        log.info(f'Reading in {infile}')
        in_df = pd.read_csv(infile)
        log.info(f'Cleaning data...')
        Y = in_df[CLASS_COLUMN]
        X = in_df.drop(CLASS_COLUMN, axis=1)

        for topics in TOPICS:
            log.info("Begin generating lda features...")
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







