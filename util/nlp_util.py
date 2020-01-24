from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import gensim.downloader as api
from nltk import WordPunctTokenizer
from util.time_util import Timer

log = logging.getLogger(__name__)

TOKENIZE_TIME_MIN = "tokenize_time_min"
VECTORIZE_TIME_MIN = "vectorize_time_min"
MODEL_SAVE_TIME_MIN = "model_save_time_min"
FILE_SAVE_TIME_MIN = "file_save_time_min"
FEATURE_TIME_MIN = "feature_time_min"

# where to save vectorizers/tokenizers
MODEL_DIR = "../models"
# where to place new feature files
OUT_DIR = "../dataset/feature_files"


def write_to_file(data: pd.DataFrame,
                  y: pd.Series,
                  feature_column,
                  description,
                  include_lda: bool,
                  lda_topics: int = 20):
    examples, features = data.shape
    # TODO: refactor this so outdir is not hardcoded
    if include_lda:
        outfile = f'{OUT_DIR}/{feature_column}-{description}-{examples}-{features}-lda{lda_topics}.csv'
    else:
        outfile = f'{OUT_DIR}/{feature_column}-{description}-{examples}-{features}-nolda.csv'
    log.info(f'writing file: {outfile}')

    # TODO: move this to feature generator later
    y["helpful_product"] = y["star_rating"].apply(lambda x: 1 if int(x) >= 3 else 0)
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
                      lda_topics: int = None,
                      min_df: float = 1,
                      max_df: float = 1.,
                      min_ngram_range=1,
                      max_ngram_range=1,
                      max_features=None
                      ) -> (pd.DataFrame, float, float):
    """
    encode feature_column using BoW

    :param x: Df with data to process
    :param y: labels
    :param feature_column: column to convert to BoW
    :param description: name/description - will be used to for description in report file
    :param lda_topics: if provided, number of LDA topics to append to dataframe
    :param min_df: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    :param max_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    :param min_ngram_range: lower boundary of the range of n-values for different n-grams to be extracted.
    :param max_ngram_range: upper boundary of the range of n-values for different n-grams to be extracted.
    :param max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
    :return:
    """
    start_time = datetime.now()
    cv = CountVectorizer(min_df=min_df,
                         max_df=max_df,
                         ngram_range=(int(min_ngram_range), int(max_ngram_range)),
                         max_features=int(max_features)
                         )
    log.debug(f'Shape of x {x.shape}')
    # there is no info because this is a Series and not DataFrame
    # log.debug(f'Info for x {x.info()}')
    cv_matrix = cv.fit_transform(x.array)
    vocab = cv.get_feature_names()
    log.info(f'vocab_length: {len(vocab)}')
    # print(f"vocab: {vocab}")
    df = pd.DataFrame(cv_matrix.toarray(), columns=vocab)

    file_start_time = datetime.now()
    outfile = write_to_file(df, y, feature_column, description, False, lda_topics)

    lda_time = 0
    lda_file_time = 0
    vectorize_file_time = 0
    if lda_topics is not None:
        lda_start_time = datetime.now()
        lda = generate_lda_feature(df, int(lda_topics))
        df = df.join(lda)

        lda_file_start_time = datetime.now()
        # get ready to write file
        outfile = write_to_file(df, y, feature_column, description, True, int(lda_topics))
        lda_file_end_time = datetime.now()

        lda_time = round((lda_file_start_time - lda_start_time).total_seconds() / 60, 2)
        lda_file_time = round((lda_file_end_time - lda_file_start_time).total_seconds() / 60, 2)
        vectorize_file_time = round((lda_start_time - file_start_time).total_seconds() / 60, 2)

    vectorize_time = round((file_start_time - start_time).total_seconds() / 60, 2)


    return outfile, df, vectorize_time, vectorize_file_time, lda_time, lda_file_time


def generate_tfidf_file(x: pd.DataFrame,
                        y: pd.Series,
                        feature_column: str,
                        description: str,
                        min_df: float = 1,
                        max_df: float = 1.,
                        min_ngram_range=1,
                        max_ngram_range=1,
                        lda_topics: int = None,
                        max_features=None
                        ) -> pd.DataFrame:
    """
    generate TFIDF encoding from feature_column

    :param x: Df with data to process
    :param y: labels
    :param feature_column: column to convert to BoW
    :param description: name/description - will be used to for description in report file
    :param lda_topics: if provided, number of LDA topics to append to dataframe
    :param min_df: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    :param max_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    :param min_ngram_range: lower boundary of the range of n-values for different n-grams to be extracted.
    :param max_ngram_range: upper boundary of the range of n-values for different n-grams to be extracted.
    :param max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
    :return:
    """
    start_time = datetime.now()
    tv = TfidfVectorizer(min_df=min_df,
                         max_df=max_df,
                         ngram_range=(int(min_ngram_range), int(max_ngram_range)),
                         max_features=int(max_features),
                         use_idf=True)
    log.debug(x.array)
    tv_matrix = tv.fit_transform(x.array)

    # TODO: save the vectorizer
    vocab = tv.get_feature_names()
    log.info(f'vocab_length: {len(vocab)}')
    df = pd.DataFrame(np.round(tv_matrix.toarray(), 2), columns=vocab)

    file_start_time = datetime.now()
    outfile = write_to_file(df, y, feature_column, description, False, lda_topics)

    lda_time = 0
    lda_file_time = 0
    vectorize_file_time = 0
    if lda_topics is not None:
        lda_start_time = datetime.now()
        lda = generate_lda_feature(df, int(lda_topics))
        df = df.join(lda)

        lda_file_start_time = datetime.now()
        # get ready to write file
        outfile = write_to_file(df, y, feature_column, description, True, lda_topics)
        lda_file_end_time = datetime.now()

        lda_time = round((lda_file_start_time - lda_start_time).total_seconds() / 60, 2)
        lda_file_time = round((lda_file_end_time - lda_file_start_time).total_seconds() / 60, 2)

        vectorize_file_time = round((lda_start_time - file_start_time).total_seconds() / 60, 2)

    vectorize_time = round((file_start_time - start_time).total_seconds() / 60, 2)

    return outfile, df, vectorize_time, vectorize_file_time, lda_time, lda_file_time



def get_average_embedding(embedding, review):
    """
    returns a list of word vectors for all words in review
    then average them to return a final vector

    :param embedding: embedding object - will be either Fasttext or Word2Vec
    :param review: review text
    :return:
    """
    log.debug(f'Getting average embedding for: [{review}]')

    wpt = WordPunctTokenizer()
    # word_vectors = [embedding.wv.get_vector(word) for word in wpt.tokenize(review)]
    word_vectors = [embedding.wv.get_vector(word) for word in wpt.tokenize(review) if word in embedding.wv.vocab]
    log.debug(f'word_vector shape [{np.shape(word_vectors)}]')

    # return average all word vectors to come up with final vector for the review
    # since we are using pre-trained embedding, we may not be able to find all the words
    if np.shape(word_vectors)[0] > 1:
        return np.average(word_vectors, axis=0)
    return None


def get_feature_df(embedding, df: pd.DataFrame) -> pd.DataFrame:
    """
    generate new feature DF

    :param embedding: this will either be a FastText or Word2Vec object
    :param df: DF with features
    :return:
    """
    f_df = pd.DataFrame()
    for index, review in df.iteritems():
        feature_vector = get_average_embedding(embedding, review)
        if feature_vector is not None:
            # turn this into dictionary so we can add it as row to DF
            feature_dict = dict(enumerate(feature_vector))
            f_df = f_df.append(feature_dict, ignore_index=True)
        if index % 10000 == 0:
            log.info(f"Generating vector for index: {index}")
    return f_df


def generate_word2vec_file(x: pd.DataFrame,
                           y: pd.Series,
                           description: str,
                           feature_column: str,
                           timer: Timer = None,
                           feature_size: int = 100,
                           window_context: int = 5,
                           min_word_count: int = 5,
                           sample: float = 0.001,
                           iterations: int = 5,
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

    if timer:
        timer.start_timer(TOKENIZE_TIME_MIN)
    documents = [wpt.tokenize(review) for review in x.array]
    if timer:
        timer.end_timer(TOKENIZE_TIME_MIN)

    if timer:
        timer.start_timer(VECTORIZE_TIME_MIN)

    # TODO: add configuraton for pre-trained or train
    # if x.shape[0] <= 50:
    w2v_model = Word2Vec(documents,
                         size=int(feature_size),
                         window=int(window_context),
                         min_count=int(min_word_count),
                         sample=sample,
                         iter=int(iterations)
                         )
    # else:
    #     log.info("Downloading pre-trained word2vec")
    #     w2v_model = api.load("word2vec-google-news-300")
    if timer:
        timer.end_timer(VECTORIZE_TIME_MIN)


    model_file = f"{MODEL_DIR}/{description}-{len(x)}-{feature_size}.model"
    log.info(f'Writing model file: {model_file}')
    if timer:
        timer.start_timer(MODEL_SAVE_TIME_MIN)
    w2v_model.save(model_file)
    if timer:
        timer.end_timer(MODEL_SAVE_TIME_MIN)

    feature_df = get_feature_df(w2v_model, x)
    return write_to_file(feature_df, y, feature_column, description, include_lda=False)


def generate_fasttext_file(x: pd.DataFrame,
                           y: pd.Series,
                           description: str,
                           feature_column: str,
                           timer: Timer = None,
                           feature_size: int = 100,
                           window_context: int = 5,
                           min_word_count: int = 5,
                           sample: float = 0.001,
                           iterations: int = 5,
                           ):
    """
    generate features using fasttext embedding

    https://radimrehurek.com/gensim/models/fasttext.html

    :param x:
    :param y:
    :param description:
    :param feature_size: Dimensionality of the word vectors
    :param window_context: The maximum distance between the current and predicted word within a sentence
    :param min_word_count: The model ignores all words with total frequency lower than this
    :param sample: The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    :param iterations: Number of iterations (epochs) over the corpus
    :return:
    """

    log.info("generating fasttext")
    log.debug(f'{x.head()}')
    wpt = WordPunctTokenizer()

    if timer:
        timer.start_timer(TOKENIZE_TIME_MIN)
    documents = [wpt.tokenize(review) for review in x.array]
    if timer:
        timer.end_timer(TOKENIZE_TIME_MIN)

    if timer:
        timer.start_timer(VECTORIZE_TIME_MIN)


    # TODO: add in configuration for pre-trained
    # if x.shape[0] <= 50:
    ft_model = FastText(documents,
                        size=int(feature_size),
                        window=int(window_context),
                        min_count=int(min_word_count),
                        sample=sample,
                        iter=int(iterations)
                        )

    # else:
    #     log.info("Download pre-trained fasttext")
    #     ft_model = FastText.load_fasttext_format('wiki.simple')
    if timer:
        timer.end_timer(VECTORIZE_TIME_MIN)

    model_file = f"{MODEL_DIR}/{description}-{len(x)}-{feature_size}.model"
    log.info(f'Writing model file: {model_file}')
    if timer:
        timer.start_timer(MODEL_SAVE_TIME_MIN)
    ft_model.save(model_file)
    if timer:
        timer.end_timer(MODEL_SAVE_TIME_MIN)

    if timer:
        timer.start_timer(FEATURE_TIME_MIN)
    feature_df = get_feature_df(ft_model, x)
    if timer:
        timer.end_timer(FEATURE_TIME_MIN)

    return write_to_file(feature_df, y, feature_column, description, include_lda=False)
