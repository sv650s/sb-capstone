from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import pandas as pd
from datetime import datetime
import numpy as np


log = logging.getLogger(__name__)
FILE_DATE_FORMAT = '%Y-%m-%d-%H'


def write_to_file(data:pd.DataFrame, feature_column, feature_name, y:pd.Series):
    date_str = datetime.now().strftime(FILE_DATE_FORMAT)
    examples, features = data.shape
    outfile = f'dataset/feature_files/{feature_column}-{feature_name}-{examples}-{features}.csv'
    log.info(f'writing file: {outfile}')
    data["star_rating"] = y
    data.to_csv(outfile, doublequote=True, index=False)


def generate_bow_file(x:pd.DataFrame,
                      y:pd.Series,
                      feature_column:str,
                      feature_name:str,
                      min_df:float=1,
                      max_df:float=1.,
                      min_ngram_range=1,
                      max_ngram_range=1,
                      max_features=None) -> pd.DataFrame:

    cv = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=(min_ngram_range, max_ngram_range))
    cv_matrix = cv.fit_transform(x.array)
    vocab = cv.get_feature_names()
    log.info(f'vocab_length: {len(vocab)}')
    # print(f"vocab: {vocab}")
    df = pd.DataFrame(cv_matrix.toarray(), columns=vocab)

    # get ready to write file
    write_to_file(df, feature_column, feature_name, y)
    return df


def generate_tfidf_file(x:pd.DataFrame,
                        y:pd.Series,
                        feature_column:str,
                        feature_name:str,
                        min_df:float=1,
                        max_df:float=1.,
                        min_ngram_range=1,
                        max_ngram_range=1,
                        max_features=None
                        ) -> pd.DataFrame:

    tv = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(min_ngram_range, max_ngram_range), max_features=max_features, use_idf=True)
    tv_matrix = tv.fit_transform(x.array)
    vocab = tv.get_feature_names()
    log.info(f'vocab_length: {len(vocab)}')
    df = pd.DataFrame(np.round(tv_matrix.toarray(), 2), columns=vocab)

    # get ready to write file
    write_to_file(df, feature_column, feature_name, y)
    return df




