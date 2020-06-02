
# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, \
    SpatialDropout1D, Flatten, LSTM, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import load_model


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from psutil import virtual_memory

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import argparse
import json


import util.keras_util as ku
import util.report_util as ru

import random

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# length of our embedding - 300 is standard
EMBED_SIZE = 300

# From EDA, we know that 90% of review bodies have 100 words or less,
# we will use this as our sequence length
MAX_SEQUENCE_LENGTH = 100


# used to fix seeds
RANDOM_SEED = 1

LABEL_COLUMN = "star_rating"
FEATURE_COLUMN = "review_body"


# TODO: set this to True and finish debugging. Currently getting  the following error
# 2020-05-20 03:19:29,495 INFO    __main__.<module> [573] - loaded json model:
# {json_loaded}
#
# Traceback (most recent call last):
#   File "train/train.py", line 574, in <module>
#     model_json_loaded = tf.keras.models.load_model(json_loaded)
#   File "/Users/vinceluk/anaconda3/envs/capstone/lib/python3.7/site-packages/tensorflow_core/python/keras/saving/save.py", line 145, in load_model
#     isinstance(filepath, h5py.File) or h5py.is_hdf5(filepath))):
#   File "/Users/vinceluk/anaconda3/envs/capstone/lib/python3.7/site-packages/h5py/_hl/base.py", line 41, in is_hdf5
#     fname = os.path.abspath(fspath(fname))
# TypeError: expected str, bytes or os.PathLike object, not dict
SAVE_JSON_FORMAT = True
# enable reduce LR on plateau callback
REDUCE_LR_ON_PLATEAU = False

# set up logging
LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
    " [%(lineno)d] - %(message)s"
logger = logging.getLogger(__name__)


def check_resources():
    """
    Check what kind of resources we have for training
    :return:
    """

    # checl to make sure we are using GPU here
    tf.test.gpu_device_name()

    # check that we are using high RAM runtime
    ram_gb = virtual_memory().total / 1e9
    logger.info('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

    # if ram_gb < 20:
    #   print('To enable a high-RAM runtime, select the Runtime → "Change runtime type"')
    #   print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
    #   print('re-execute this cell.')
    # else:
    #   print('You are using a high-RAM runtime!')


def fix_seed(seed: int):
    logger.info(f"Fixing rando seed to {seed}")

    # fix random seeds
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_data(data_file: str, feature_column:str, label_column: str):
    """
    Loads data file
    OHE labelcolumn
    split features and labels into train and test set

    :param data_file:
    :param feature_column:
    :param label_column:
    :return:
    """
    logger.info(f'Reading data file: {data_file} feature_column: {feature_column} label_column: {label_column}')

    df = pd.read_csv(data_file)

    # drop any rows with empty feature columns that we may have missed
    logger.info("Dropping rows with empty features...")
    df.dropna(subset = [feature_column, label_column], inplace = True)

    reviews = df[feature_column]
    ratings = df[label_column]

    # pre-process our lables
    # one hot encode our star ratings since Keras/TF requires this for the labels
    y = OneHotEncoder().fit_transform(ratings.values.reshape(len(ratings), 1)).toarray()


    # split our data into train and test sets
    reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, y, random_state=1)

    logger.debug(f'features dimensions - train {np.shape(reviews_train)} test {np.shape(reviews_test)}')
    logger.debug(f'label dimensions - train {np.shape(y_train)} test {np.shape(y_test)}')

    return reviews_train, reviews_test, y_train, y_test, ratings

def preprocess_data(feature_train, feature_test, embedding_file: str, missing_words_file: str):
    """
    Tokenize text and convert features into embedding matrix

    Pre-trained embedding will be converted into an index file in the same directory as embedding file
    This index file will be loaded instead of the embedding file to save time

    :param feature_train: training feature set
    :param feature_test: test feature set
    :param embedding_file: full path name of pre-trained embedding file
    :param missing_words_file: full path name to store missing words csv file
    :return:
        X_train, X_test, t, embedding_matrix
    """
    logger.info(f'feature train dimensions {np.shape(feature_train)} feature test dimensions {np.shape(feature_test)}')

    logger.info("Tokenizing features...")
    # Pre-process our features (review body)
    t = Tokenizer(oov_token="<UNK>")
    # fit the tokenizer on the documents
    t.fit_on_texts(feature_train)
    # tokenize both our training and test data
    train_sequences = t.texts_to_sequences(feature_train)
    test_sequences = t.texts_to_sequences(feature_test)

    logger.info("Vocabulary size={}".format(len(t.word_counts)))
    logger.info("Number of Documents={}".format(t.document_count))


    # pad our reviews to the max sequence length
    X_train = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    logger.info(f'Train review vectors shape: {np.shape(X_train)} Test review vectors shape: {np.shape(X_test)}')

    """## Load our pre-trained embedding

    embedding_index will be a map where key == word, value == the embedding vector
    """

    embedding_dir = os.path.dirname(embedding_file)
    embedding_index_file = f'{embedding_dir}/glove.840B.300d-embedding_index'

    embedding_index = {}

    logger.info("Creating embedding index...")
    if os.path.exists(f'{embedding_index_file}.npy'):
      logger.info(f'Loading {embedding_index_file}.npy')
      embedding_index = np.load(f'{embedding_index_file}.npy',
                                 allow_pickle = True).item()
    else:
      logger.info(f'{embedding_index_file} does not exist. Indexing words from {embedding_file}...')

      with open(embedding_file) as f:
          for line in f:
              word, coefs = line.split(maxsplit=1)
              coefs = np.fromstring(coefs, 'f', sep=' ')
              embedding_index[word] = coefs
      logger.info(f'Saving embedding index to {embedding_index_file}...')
      np.save(embedding_index_file, embedding_index)

    logger.debug(f'embedding_index type {type(embedding_index)} shape {np.shape(embedding_index)}')
    logger.info('Found %s word vectors.' % len(embedding_index))



    """## Create Embedding Matrix based on our tokenizer

    For every word in our vocabulary, we will look up the embedding vector and add the it to our embedding matrix

    The matrix will be passed in as weights in our embedding layer later

    If there is word that does not exist in the pre-trained embedding vocabulary, we will leave the weights as 0 vector and save off the word into a CSV file later for analysis
    """

    # this is a map with key == word, value == index in the vocabulary
    word_index = t.word_index
    logger.info(f'word_index length: {len(word_index)}')

    # start with a matrix of 0's
    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
    logger.debug(f'embedding_matrix shape: {np.shape(embedding_matrix)}')

    # if a word doesn't exist in our vocabulary, let's save it off
    logger.info("Creating embedding matrix from embedding index...")
    missing_words = []
    for word, i in word_index.items():
        # logger.info(f'word: {word} i: {i}')
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None and np.shape(embedding_vector)[0] == EMBED_SIZE:
            # words not found in embedding index will be all-zeros.
            # logger.info(f'i: {i} embedding_vector shape: {np.shape(embedding_vector)}')
            embedding_matrix[i] = embedding_vector
        else:
          missing_words.append(word)

    logger.info(f'Number of missing words from our vocabulary: {len(missing_words)}')

    """Save off our missing words into a csv file so we can analyze this later"""

    # save missing words into a file so we can analyze it later
    missing_words_df = pd.DataFrame(missing_words)
    # sort missing words so the output is always the same
    print(missing_words_df.columns)
    missing_words_df = missing_words_df.sort_values(0)
    logger.info("Saving missing words file...")
    missing_words_df.to_csv(missing_words_file, index=False)

    return X_train, X_test, t, embedding_matrix



if __name__ == "__main__":

    start_time = datetime.now()

    check_resources()
    fix_seed(RANDOM_SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--learning_rate", help="Optimizer learning rate. Default = 0.001", default=0.001)
    parser.add_argument("-b", "--batch_size", help="Training batch size. Default = 32", default=32)
    parser.add_argument("-c", "--cells", help="Number of LSTM cells. Default = 128", default=128)
    parser.add_argument("-d", "--dropout_rate", help="dropout rate. Default 0",
                        default=0)
    parser.add_argument("-e", "--epochs", help="Max number epochs. Default = 20", default=20)
    parser.add_argument("-i", "--input_dir", help="input directory. Default /storage/data",
                        default="/storage/data")
    parser.add_argument("-l", "--loglevel", help="log level", default="INFO")
    parser.add_argument("-m", "--train_embeddings",
                        help="set this flag to make enbedding layer trainable. Default False",
                        default=False,
                        action="store_true")
    parser.add_argument("-n", "--bidirectional",
                        help="label column. Default star_rating",
                        default=False,
                        action="store_true")
    parser.add_argument("-o", "--output_dir", help="output directory. Default /artifacts",
                        default="/artifacts")
    parser.add_argument("-p", "--patience", help="patience. Default = 4", default=4)
    parser.add_argument("-r", "--recurrent_dropout_rate", help="recurrent dropout rate. NOTE: will not be able to " \
                                                               "cuDNN if this is set. Default 0",
                        default=0)
    parser.add_argument("-t", "--network_type", help="network type - ie, LSTM or GRU (required)", default = None)
    parser.add_argument("-s", "--resume_training_file", help="path to load model file to resume training", default = None)
    parser.add_argument("-u", "--unbalanced_class_weights",
                        help="do not balance class weights for training",
                        default=False,
                        action="store_true")
    parser.add_argument("-v", "--model_version", help="Specify model version. Default = 1", default=1)

    # POSITIONAL ARGUMENTS
    parser.add_argument("sample_size",
                        help="Sample size (ie, 50k, test)",
                        type=str)

    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    logger = logging.getLogger(__name__)






    input_dir = args.input_dir
    output_dir = args.output_dir
    sample_size = args.sample_size
    network_type = args.network_type


    if network_type is None:
        print("ERROR: -t/--network_type is required")
        exit(1)
    elif network_type not in ["LSTM", "GRU"]:
        print("ERROR: network_type must be LSTM or GRU")
        exit(1)

    cells = int(args.cells)
    epochs  = int(args.epochs)
    batch_size = int(args.batch_size)
    patience = int(args.patience)
    learning_rate = float(args.learning_rate)
    model_version = args.model_version

    dropout_rate = float(args.dropout_rate)
    recurrent_dropout_rate = float(args.recurrent_dropout_rate)

    bidirectional = args.bidirectional
    balance_class_weights = not args.unbalanced_class_weights
    train_embeddings = args.train_embeddings

    resume_training_file = args.resume_training_file
    if resume_training_file is not None and not os.path.exists(resume_training_file):
        raise FileNotFoundError(f"File not found {resume_training_file}")


    data_dir = f'{input_dir}/amazon_reviews'
    embeddings_dir = f'{input_dir}/embeddings'
    reports_dir = f'{output_dir}/reports'
    models_dir = f'{output_dir}/models'

    debug = False
    if sample_size == "test":
        logger.info("Running in DEBUG mode")
        debug = True

    # process argument
    if debug:
        loglevel = logging.DEBUG
    elif args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    else:
        loglevel = logging.WARN

    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    logger = logging.getLogger(__name__)

    if bidirectional:
        bidirectional_name = "bi"
    else:
        bidirectional_name = ""

    if balance_class_weights:
        balanced_name = "B"
    else:
        balanced_name = ""

    model_name = f"{bidirectional_name}{network_type}{balanced_name}{cells}"
    DESCRIPTION = f"1 Layer {cells} {model_name} Units, Dropout {dropout_rate}, Recurrent Dropout {recurrent_dropout_rate}, Batch Size {batch_size}, Learning Rate {learning_rate}"
    architecture = f"1x{cells}"
    FEATURE_SET_NAME = "glove_with_stop_nonlemmatized"
    # TODO: add in sampling options


    REPORT_FILE = f"paperspace-{model_name}-" \
        f"{architecture}-" \
        f"dr{ku.get_decimal_str(dropout_rate)}-" \
        f"rdr{ku.get_decimal_str(recurrent_dropout_rate)}-" \
        f"batch{batch_size}-" \
        f"lr{ku.get_decimal_str(learning_rate)}-" \
        f"{FEATURE_SET_NAME}-" \
        f"sampling_none-" \
        f"{FEATURE_COLUMN}-" \
            "report.csv"


    # print available devices
    logger.info(f"\n\nAvailable Devices for Tensorflow:\n{device_lib.list_local_devices()}\nEnd Devise List")
    # number of GPU's available
    available_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    logger.info(f"\nNum GPUs Available: {available_gpus}")
    # print tensorflow placement - need to create a small model and run to log this
    # comment out for now - too much logs
    tf.debugging.set_log_device_placement(True)

    if not debug and available_gpus < 1:
        logger.error("Unable to find GPU for training. Exiting")
        exit(1)



    STORAGE_DIR = "/storage"
    if debug:
      data_file = f'{data_dir}/amazon_reviews_us_Wireless_v1_00-test-with_stop_nonlemmatized-preprocessed.csv'
      model_name = f'test-{model_name}'
      MISSING_WORDS_FILE = f'{reports_dir}/glove_embedding-missing_words-test.csv'
      ku.ModelWrapper.set_report_filename(f'test-{REPORT_FILE}')

      # for local testing only
      STORAGE_DIR = "./storage"
      if not os.path.exists(STORAGE_DIR):
          logger.info(f"Creating {STORAGE_DIR}")
          os.mkdir(STORAGE_DIR)
    else:
      data_file = f"{data_dir}/amazon_reviews_us_Wireless_v1_00-{sample_size}-with_stop_nonlemmatized-preprocessed.csv"
      MISSING_WORDS_FILE = f'{reports_dir}/glove_embedding-missing_words-{sample_size}.csv'
      ku.ModelWrapper.set_report_filename(REPORT_FILE)


    EMBEDDING_FILE = f'{embeddings_dir}/glove.840B.300d.txt'





    ##### validate that we have the correct directories before we start
    if os.path.exists(input_dir):
        if not os.path.exists(f'{embeddings_dir}'):
            os.mkdir(f'{embeddings_dir}')
    else:
        logger.error(f'ERROR: {input_dir} does not exist')
        exit(1)

    if os.path.exists(output_dir):
        if not os.path.exists(f'{reports_dir}'):
            logger.info(f'{reports_dir} missing. Creating...')
            os.mkdir(f'{reports_dir}')
        if not os.path.exists(f'{models_dir}'):
            logger.info(f'{models_dir} missing. Creating...')
            os.mkdir(f'{models_dir}')
    else:
        logger.error(f'ERROR: {output_dir} does not exist')
        exit(1)

    if not os.path.exists(STORAGE_DIR):
        logger.warning(f'WARNING: {STORAGE_DIR} does not exist')


    reviews_train, reviews_test, y_train, y_test, ratings = load_data(data_file, FEATURE_COLUMN, LABEL_COLUMN)

    X_train, X_test, t, embedding_matrix = preprocess_data(reviews_train,
                                                           reviews_test,
                                                           EMBEDDING_FILE,
                                                           MISSING_WORDS_FILE)

    vocab_size = len(t.word_index)+1


    logger.debug(f'y_train: {y_train[:5]} ratings {ratings[:5]}')
    weights = compute_class_weight('balanced', np.arange(1, 6), ratings)
    weights_dict = {i: weights[i] for i in np.arange(0, len(weights))}
    logger.info(f'class weights: {weights}')
    logger.info(f'class weights_dict: {weights_dict}')



    if network_type == "LSTM":
        mw = ku.LSTM1LayerModelWrapper(
                                dimension= cells, # LSTM dim - LSTM1LyerModelWrapper
                                 dropout_rate = dropout_rate, # dropout rate - LSTM1LyerModelWrapper
                                 recurrent_dropout_rate = recurrent_dropout_rate, # recurrent dropout rate - LSTM1LyerModelWrapper
                                 bidirectional = bidirectional, # bidirectional - LSTM1LyerModelWrapper
                                 vocab_size = vocab_size,       # vocab size - EmbeddingModelWrapper
                                 max_sequence_length = MAX_SEQUENCE_LENGTH, # max sequence length - EmbeddingModelWrapper
                                 embed_size = EMBED_SIZE, # embed size - EmbeddingModelWrapper
                                train_embeddings  =  train_embeddings, # trainable embedding - EmbeddingModelWrapper
            embedding_matrix=embedding_matrix,  # pre-trained embedding matrix - EmbeddingModelWrapper
            model_name = model_name, # model name - ModelWrapper
                                architecture = architecture, # architecture - ModelWrapper
                                feature_set_name = FEATURE_SET_NAME, # feature_set_name - ModelWrapper
                                label_column = LABEL_COLUMN, # label_column - ModelWrapper
                                feature_column = FEATURE_COLUMN, # feature_column - ModelWrapper
                                data_file = data_file, # data file - ModelWrapper
                                sample_size_str = sample_size, # sample size
                                tokenizer = t, # tokenizer - ModelWrapper
                                description = DESCRIPTION, #description - ModelWrapper
                                optimizer_name = "Adam", # string optimizer name
                                learning_rate = learning_rate, # learning rate - ModelWrapper
                                # TODO: this should be in fit instead but need it to define name for the
                                # checkpoint location - move later
                                batch_size = batch_size, # batch size - ModelWrapper
                                model_version= model_version, # model version - ModelWrapper
                                save_dir = output_dir, # where to save outputs - ModelWrapper
                                load_model_file= resume_training_file # load model from file - ModelWrapper
        )
    elif network_type == "GRU":
        mw = ku.GRU1LayerModelWrapper(
            dimension= cells, # LSTM dim - LSTM1LyerModelWrapper
            dropout_rate = dropout_rate, # dropout rate - LSTM1LyerModelWrapper
            recurrent_dropout_rate = recurrent_dropout_rate, # recurrent dropout rate - LSTM1LyerModelWrapper
            bidirectional = bidirectional, # bidirectional - LSTM1LyerModelWrapper
            vocab_size = vocab_size,       # vocab size - EmbeddingModelWrapper
            max_sequence_length = MAX_SEQUENCE_LENGTH, # max sequence length - EmbeddingModelWrapper
            embed_size = EMBED_SIZE, # embed size - EmbeddingModelWrapper
            train_embeddings  =  train_embeddings, # trainable embedding - EmbeddingModelWrapper
            embedding_matrix=embedding_matrix,  # pre-trained embedding matrix - EmbeddingModelWrapper
            model_name = model_name, # model name - ModelWrapper
            architecture = architecture, # architecture - ModelWrapper
            feature_set_name = FEATURE_SET_NAME, # feature_set_name - ModelWrapper
            label_column = LABEL_COLUMN, # label_column - ModelWrapper
            feature_column = FEATURE_COLUMN, # feature_column - ModelWrapper
            data_file = data_file, # data file - ModelWrapper
            sample_size_str = sample_size, # sample size
            tokenizer = t, # tokenizer - ModelWrapper
            description = DESCRIPTION, #description - ModelWrapper
            optimizer_name = "Adam", # string optimizer name
            learning_rate = learning_rate, # learning rate - ModelWrapper
            # TODO: this should be in fit instead but need it to define name for the
            # checkpoint location - move later
            batch_size = batch_size, # batch size - ModelWrapper
            model_version= model_version, # model version - ModelWrapper
            save_dir = output_dir, # where to save outputs - ModelWrapper
            load_model_file= resume_training_file # load model from file - ModelWrapper
        )


    mw.add("environment", "paperspace")
    mw.add("patience", patience)

    callbacks = []

    early_stop = EarlyStopping(monitor = 'val_loss',
                               patience = patience,
                               verbose = 1,
                               restore_best_weights = True)
    callbacks.append(early_stop)

    if REDUCE_LR_ON_PLATEAU:
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                      restore_best_weights = True,
                                      factor = 0.5,
                                      verbose = 1,
                                      patience = 2)
        callbacks.append(reduce_lr)



    if os.path.exists(STORAGE_DIR):
        storage_model_filepath = f'{STORAGE_DIR}/{ku.ModelWrapper.models_dir}/{mw._get_saved_file_basename()}'
        if not os.path.exists(storage_model_filepath):
            os.makedirs(storage_model_filepath, exist_ok = True)
        logger.info(f"Adding {storage_model_filepath} checkpoint callback...")
        checkpoint_storage = tf.keras.callbacks.ModelCheckpoint(
            filepath = storage_model_filepath,
            verbose = 1,
            save_weights_only = True,
            monitor = 'val_loss',
            save_freq = 'epoch',
            save_best_only = True)
        callbacks.append(checkpoint_storage)



    network_history = mw.fit(X_train,
                             y_train,
                             epochs=epochs,
                             verbose=1,
                             validation_split=0.2,
                             balance_class_weights=balance_class_weights,
                             callbacks=callbacks)

    mw.evaluate(X_test, y_test)

    logger.info("Train Accuracy: %.2f%%" % (mw.train_scores[1]*100))
    logger.info("Test Accuracy: %.2f%%" % (mw.test_scores[1]*100))

    logger.info("\nConfusion Matrix")
    logger.info(mw.test_confusion_matrix)
    
    logger.info("\nClassification Report")
    logger.info(mw.test_classification_report)

    custom_score = ru.calculate_metric(mw.test_crd)
    logger.info(f'Custom Score: {custom_score}')

    """**Save off various files**"""

    mw.save(append_report=True)


    ###############################################
    # Save models in /storage on paperspace so we can access the models later
    ###############################################
    # TODO: implement storing model to paperspace
    if os.path.exists(STORAGE_DIR):

        storage_model_filepath_with_version = f"{storage_model_filepath}/{model_version}"
        # if not os.path.exists(storage_model_filepath_with_version):
        #     os.makedirs(storage_model_filepath_with_version, exist_ok= True)
        # logger.info(f"Saving model to SavedModel format {storage_model_filepath_with_version}")
        # mw.model.save(storage_model_filepath_with_version)

        if SAVE_JSON_FORMAT:

            model_json_filepath = f"{storage_model_filepath}/{os.path.basename(mw.model_json_file)}"
            print(f"Saving json config file: {model_json_filepath}")
            model_json = mw.model.to_json()
            with open(model_json_filepath, 'w') as json_file:
                json_file.write(model_json)

            with open(model_json_filepath, 'r') as file:
                logger.info(f'loaded model json:\n{json.load(file)}')

            weights_json_filepath = f"{storage_model_filepath}/{os.path.basename(mw.weights_file)}"
            print(f"Saving weights file: {weights_json_filepath}")
            mw.model.save_weights(weights_json_filepath,
                                    save_format=None)

            files = os.listdir(storage_model_filepath)
            logger.info(f"Files in {storage_model_filepath}:\n{files}")


    else:
        logger.error(f"{STORAGE_DIR} not found")


    """# Test That Our Models Saved Correctly"""

    # TODO: uncomment this out for future
    # logger.info("\nReloading model for testing...")
    # model_loaded = load_model(mw.model_file)
    # scores = model_loaded.evaluate(X_test, y_test, verbose=1)
    # accuracy = scores[1] * 100
    # logger.info("Loaded Model Accuracy: %.2f%%" % (accuracy))


    # Loading model from json files
    logger.info("\bReloading model from JSON for testing...")
    logger.info(f"json model file: {mw.model_json_file}")
    with open(mw.model_json_file) as file:
        # example from model_builder
        # with open(model_json_file) as json_file:
        #     json_config = json_file.read()
        # model = keras.models.model_from_json(json_config)
        json_loaded = file.read()
        model_json_loaded = tf.keras.models.model_from_json(json_loaded)

        # didn't work
        # json_loaded = json.load(file)
        # model_json_loaded = tf.keras.models.load_model(json_loaded)
        logger.info(f"Loaded SavedMode:\n{model_json_loaded.summary()}")

        logger.info(f"json weights file: {mw.weights_file}")
        model_json_loaded.load_weights(mw.weights_file)

        model_json_loaded.compile(loss="categorical_crossentropy",
                      optimizer=eval("Adam")(learning_rate = learning_rate),
                      metrics=["categorical_accuracy"])

        scores_json = model_json_loaded.evaluate(X_test, y_test, verbose=1)
        accuracy_json = scores_json[1] * 100
        logger.info("Loaded JSON Model Accuracy: %.2f%%" % (accuracy_json))

        # this takes too long for real models
        y_predict = model_json_loaded.predict(X_test)
        y_predict_unencoded = ku.unencode(y_predict)
        y_test_unencoded = ku.unencode(y_test)

        # classification report
        logger.info(classification_report(y_test_unencoded, y_predict_unencoded))

        # confusion matrix
        logger.info(confusion_matrix(y_test_unencoded, y_predict_unencoded))

    # Loading model from SavedModel format
    # logger.info("\nReloading model for testing...")
    # savedmodel_dir = f'{os.path.dirname(mw.model_file)}/{model_version}'
    # logger.info(f"SavedModel dir: {savedmodel_dir}")
    # loaded_savedmodel = tf.keras.models.load_model(savedmodel_dir)
    # logger.info(f"Loaded SavedModel:\n{loaded_savedmodel.summary()}")

    # scores_savedmodel = loaded_savedmodel.evaluate(X_test, y_test, verbose=1)
    # accuracy_savedmodel = scores_savedmodel[1] * 100
    # logger.info("Loaded SavedModel Accuracy: %.2f%%" % (accuracy_savedmodel))


    end_time = datetime.now()
    logger.info(f'Finished training model:\n{mw}')
    logger.info("Accuracy: %.2f%%" % (accuracy_json))
    logger.info("Custom Score: %.2f%%" % (custom_score))
    logger.info(f'Star Time: {start_time}')
    logger.info(f'End Time: {end_time}')
    logger.info(f'Total Duration: {round((end_time - start_time).total_seconds() / 60, 2)} mins')
