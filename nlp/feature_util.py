from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from nltk import WordPunctTokenizer
from util.program_util import Timer
from nltk import WordPunctTokenizer

log = logging.getLogger(__name__)

TOKENIZE_TIME_MIN = "tokenize_time_min"
VECTORIZE_TIME_MIN = "vectorize_time_min"
MODEL_SAVE_TIME_MIN = "model_save_time_min"
FILE_SAVE_TIME_MIN = "file_save_time_min"
FEATURE_TIME_MIN = "feature_time_min"


def write_to_file(data: pd.DataFrame,
                  y: pd.Series,
                  feature_column,
                  description,
                  include_lda: bool,
                  lda_topics: int = 20):
    examples, features = data.shape
    if include_lda:
        outfile = f'dataset/feature_files/{feature_column}-{description}-{examples}-{features}-lda{lda_topics}.csv'
    else:
        outfile = f'dataset/feature_files/{feature_column}-{description}-{examples}-{features}-nolda.csv'
    log.info(f'writing file: {outfile}')
    # TODO: move this to feature generator later
    y["helpful_product"] = y["star_rating"].apply(lambda x: "Yes" if x >= 3 else "No")
    data = data.join(y)
    data.to_csv(outfile, doublequote=True, index=False)
    return outfile


def generate_lda_feature(x, topic) -> pd.DataFrame:
    log.info(f"Generating lda features from x:{x.shape} topics:{topic}")
    lda = LatentDirichletAllocation(n_components=topic, max_iter=10, random_state=0)
    dt_matrix = lda.fit_transform(x)
    return pd.DataFrame(dt_matrix)


def generate_bow_file(x: pd.DataFrame,
                      y: pd.Series,
                      feature_column: str,
                      description: str,
                      lda_topics: int = 20,
                      min_df: float = 1,
                      max_df: float = 1.,
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


def generate_tfidf_file(x: pd.DataFrame,
                        y: pd.Series,
                        feature_column: str,
                        description: str,
                        lda_topics: int = 20,
                        min_df: float = 1,
                        max_df: float = 1.,
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
    # TODO: save the vectorizer
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


def get_review_vector(model, review):
    # returns a list of word vectors for all words im review
    wpt = WordPunctTokenizer()
    word_vectors = [model.wv.get_vector(word) for word in wpt.tokenize(review)]
    #     print(len(word_vectors))
    # average all word vectors to come up with final vector for the review
    return np.average(word_vectors, axis=0)


# generate new feature DF
def get_feature_df(model, df: pd.DataFrame) -> pd.DataFrame:
    f_df = pd.DataFrame()
    for index, review in df.iteritems():
        feature_vector = get_review_vector(model, review)
        # turn this into dictionary so we can add it as row to DF
        feature_dict = dict(enumerate(feature_vector))
        f_df = f_df.append(feature_dict, ignore_index=True)
        if index % 10000 == 0:
            log.info(f"Generating vector for index: {index}")
    return f_df


def generate_word2vec_file(x: pd.DataFrame,
                           y: pd.Series,
                           description: str,
                           feature_size: int,
                           window_context: int,
                           min_word_count: int,
                           sample: float,
                           iterations: int,
                           feature_column: str,
                           timer: Timer
                           ):
    """
    generate features using word2vec
    :param x:
    :param y:
    :param description:
    :param feature_size:
    :param window_context:
    :param min_word_count:
    :param sample:
    :param iterations:
    :return:
    """
    log.info("generating word2vec")
    log.debug(f'{x.head()}')
    wpt = WordPunctTokenizer()

    timer.start_timer(TOKENIZE_TIME_MIN)
    documents = [wpt.tokenize(review) for review in x.array]
    timer.end_timer(TOKENIZE_TIME_MIN)

    timer.start_timer(VECTORIZE_TIME_MIN)
    w2v_model = Word2Vec(documents,
                         size=feature_size,
                         window=window_context,
                         min_count=min_word_count,
                         sample=sample,
                         iter=iterations
                         )
    timer.end_timer(VECTORIZE_TIME_MIN)

    model_file = f"models/{description}-{len(x)}-{feature_size}.model"
    log.info(f'Writing model file: {model_file}')
    timer.start_timer(MODEL_SAVE_TIME_MIN)
    w2v_model.save(model_file)
    timer.end_timer(MODEL_SAVE_TIME_MIN)

    feature_df = get_feature_df(w2v_model, x)
    return write_to_file(feature_df, y, feature_column, description, include_lda=False)


def generate_fasttext_file(x: pd.DataFrame,
                           y: pd.Series,
                           description: str,
                           feature_size: int,
                           window_context: int,
                           min_word_count: int,
                           sample: float,
                           iterations: int,
                           feature_column: str,
                           timer: Timer
                           ):
    """
    generate features using word2vec
    :param x:
    :param y:
    :param description:
    :param feature_size:
    :param window_context:
    :param min_word_count:
    :param sample:
    :param iterations:
    :return:
    """
    log.info("generating fasttext")
    log.debug(f'{x.head()}')
    wpt = WordPunctTokenizer()

    timer.start_timer(TOKENIZE_TIME_MIN)
    documents = [wpt.tokenize(review) for review in x.array]
    timer.end_timer(TOKENIZE_TIME_MIN)

    timer.start_timer(VECTORIZE_TIME_MIN)
    ft_model = FastText(documents,
                        size=feature_size,
                        window=window_context,
                        min_count=min_word_count,
                        sample=sample,
                        iter=iterations
                        )
    timer.end_timer(VECTORIZE_TIME_MIN)

    model_file = f"models/{description}-{len(x)}-{feature_size}.model"
    log.info(f'Writing model file: {model_file}')
    timer.start_timer(MODEL_SAVE_TIME_MIN)
    ft_model.save(model_file)
    timer.end_timer(MODEL_SAVE_TIME_MIN)

    timer.start_timer(FEATURE_TIME_MIN)
    feature_df = get_feature_df(ft_model, x)
    timer.end_timer(FEATURE_TIME_MIN)

    return write_to_file(feature_df, y, feature_column, description, include_lda=False)
