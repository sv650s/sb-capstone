from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime


log = logging.getLogger(__name__)


def write_to_file(data:pd.DataFrame,
                  y:pd.Series,
                  feature_column,
                  description,
                  include_lda:bool,
                  lda_topics:int):
    examples, features = data.shape
    if include_lda:
        outfile = f'dataset/feature_files/{feature_column}-{description}-{examples}-{features}-lda{lda_topics}.csv'
    else:
        outfile = f'dataset/feature_files/{feature_column}-{description}-{examples}-{features}-nolda.csv'
    log.info(f'writing file: {outfile}')
    data = data.join(y)
    data.to_csv(outfile, doublequote=True, index=False)


def generate_lda_feature(x, topic) -> pd.DataFrame:
    log.info(f"Generating lda features from x:{x.shape} topics:{topic}")
    lda = LatentDirichletAllocation(n_components=topic, max_iter=10, random_state=0)
    dt_matrix = lda.fit_transform(x)
    return pd.DataFrame(dt_matrix)


def generate_bow_file(x:pd.DataFrame,
                      y:pd.Series,
                      feature_column:str,
                      description:str,
                      lda_topics:int=20,
                      min_df:float=1,
                      max_df:float=1.,
                      min_ngram_range=1,
                      max_ngram_range=1,
                      max_features=None
                      ) -> (pd.DataFrame, float, float):

    start_time = datetime.now()
    cv = CountVectorizer(min_df=min_df,
                         max_df=max_df,
                         ngram_range=(min_ngram_range, max_ngram_range),
                         max_features=max_features)
    cv_matrix = cv.fit_transform(x.array)
    vocab = cv.get_feature_names()
    log.info(f'vocab_length: {len(vocab)}')
    # print(f"vocab: {vocab}")
    df = pd.DataFrame(cv_matrix.toarray(), columns=vocab)

    file_start_time = datetime.now()
    write_to_file(df, y, feature_column, description, False, lda_topics)

    lda_start_time = datetime.now()
    lda = generate_lda_feature(df, int(lda_topics))
    df = df.join(lda)

    lda_file_start_time = datetime.now()
    # get ready to write file
    write_to_file(df, y, feature_column, description, True, lda_topics)
    lda_file_end_time = datetime.now()

    vectorize_time = round((file_start_time - start_time).total_seconds() / 60, 2)
    vectorize_file_time = round((lda_start_time - file_start_time).total_seconds() / 60, 2)
    lda_time = round((lda_file_start_time - lda_start_time).total_seconds() / 60, 2)
    lda_file_time = round((lda_file_end_time - lda_file_start_time).total_seconds() / 60, 2)

    return df, vectorize_time, vectorize_file_time, lda_time, lda_file_time


def generate_tfidf_file(x:pd.DataFrame,
                        y:pd.Series,
                        feature_column:str,
                        description:str,
                        lda_topics:int=20,
                        min_df:float=1,
                        max_df:float=1.,
                        min_ngram_range=1,
                        max_ngram_range=1,
                        max_features=None
                        ) -> pd.DataFrame:

    start_time = datetime.now()
    tv = TfidfVectorizer(min_df=min_df,
                         max_df=max_df,
                         ngram_range=(min_ngram_range, max_ngram_range),
                         max_features=max_features,
                         use_idf=True)
    tv_matrix = tv.fit_transform(x.array)
    vocab = tv.get_feature_names()
    log.info(f'vocab_length: {len(vocab)}')
    df = pd.DataFrame(np.round(tv_matrix.toarray(), 2), columns=vocab)

    file_start_time = datetime.now()
    write_to_file(df, y, feature_column, description, False, lda_topics)

    lda_start_time = datetime.now()
    lda = generate_lda_feature(df, int(lda_topics))
    df = df.join(lda)

    lda_file_start_time = datetime.now()
    # get ready to write file
    write_to_file(df, y, feature_column, description, True, lda_topics)
    lda_file_end_time = datetime.now()

    vectorize_time = round((file_start_time - start_time).total_seconds() / 60, 2)
    vectorize_file_time = round((lda_start_time - file_start_time).total_seconds() / 60, 2)
    lda_time = round((lda_file_start_time - lda_start_time).total_seconds() / 60, 2)
    lda_file_time = round((lda_file_end_time - lda_file_start_time).total_seconds() / 60, 2)

    return df, vectorize_time, vectorize_file_time, lda_time, lda_file_time




