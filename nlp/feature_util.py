from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import pandas as pd
import numpy as np


log = logging.getLogger(__name__)


def write_to_file(data:pd.DataFrame, feature_column, description, y:pd.Series):
    examples, features = data.shape
    outfile = f'dataset/feature_files/{feature_column}-{description}-{examples}-{features}.csv'
    log.info(f'writing file: {outfile}')
    data["star_rating"] = y
    data.to_csv(outfile, doublequote=True, index=False)


def generate_bow_file(x:pd.DataFrame,
                      y:pd.Series,
                      feature_column:str,
                      description:str,
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
    write_to_file(df, feature_column, description, y)
    return df


def generate_tfidf_file(x:pd.DataFrame,
                        y:pd.Series,
                        feature_column:str,
                        description:str,
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
    write_to_file(df, feature_column, description, y)
    return df




