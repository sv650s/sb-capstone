from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, \
    SpatialDropout1D, Flatten, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from abc import ABC, ABCMeta, abstractmethod, abstractproperty, abstractclassmethod, abstractstaticmethod

import util.file_util as fu
import json
import logging
import os
from datetime import datetime
import pickle
from functools import singledispatch



from util.metric_util import calculate_roc_auc

DATE_FORMAT = '%Y-%m'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

log = logging.getLogger(__name__)

def get_decimal_str(flt):
  """
  gets the parts after decimal for a float as a string

  will use this to parse DL parameters to generate filenames
  """
  return str(flt).split(".")[1]


def unencode(input):
    """
    unencode input
    input is assumed to be probability returned by model - ie, columns are the classes, values in each row is the probability of each class
    unencoded output would be a 1D array with each element containing the class index with highest probability

    :param input: np array or pandas DataFrame - should be 2D
    :return:
    """
    if isinstance(input, np.ndarray):
        input_df = pd.DataFrame(input)
    elif isinstance(input, pd.DataFrame):
        input_df = input

    ret =  [row.idxmax() + 1 for index, row in input_df.iterrows()]
    log.debug(f'Unencoded: {ret[:5]}')
    return ret


# TODO: move this method into ModelWrapper so we don't have to set sampling_type
# explicitly - we should be able infer this using type(sampler).__name__.lower()
def preprocess_file(data_df,
                    feature_column,
                    label_column,
                    report_dir,
                    test_size = None,
                    random_state = 1,
                    max_sequence_length = 100,
                    use_oov_token=True,
                    sampler = None
                    ):
    """
    Preprocessing data file and create the right inputs for Keras models
        Generally using this for more advanced models like GRU, LSTM, CNN as we are embedding as part of this

        Features:
        * tokenize
        * pad features into sequence
        Labels:
        * one hot encoder

        split between training and testing

    :param data_df: DF with both features and label
    :param feature_column: string name of feature column
    :param label_column: string name of lable column
    :param report_dir: string - where to save sampling histogram
    :param max_sequence_length: maximum number of words to keep in the review
    :param use_oov_token: Default True. Use a out of vocabulary token for tokenizer
    :param sampler: imblearn sampler must have fit_resample function
    :return: X_train, X_test, y_train, y_test, tokenizer
    """
    labels = data_df[label_column]
    features = data_df[feature_column]

    # Pre-process our features (review body)
    if use_oov_token:
        t = Tokenizer(lower=True, oov_token="<UNK>")
    else:
        t = Tokenizer(lower=True)

    # fit the tokenizer on the documents
    t.fit_on_texts(features)
    # tokenize both our training and test data
    features_sequences = t.texts_to_sequences(features)

    # pad our reviews to the max sequence length
    features_padded = sequence.pad_sequences(features_sequences, maxlen=max_sequence_length)

    print("Vocabulary size={}".format(len(t.word_index)))
    print("Number of Documents={}".format(t.document_count))


    # split our data into train and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train_unencoded, y_test_unencoded = train_test_split(features_padded,
                                                                            labels,
                                                                            test_size=test_size,
                                                                            random_state=random_state)
    print(f'Training X type {type(X_train)} y type {type(y_train_unencoded)}')
    print(f'Training X shape {X_train.shape} y shape {y_train_unencoded.shape}')
    print(f'Test X shape {X_test.shape} y shape {y_test_unencoded.shape}')


    if sampler is not None:
        print(f'Start sampling with {sampler.__class__.__name__}')
        X_train, y_train_unencoded = sampler.fit_resample(X_train, y_train_unencoded)
        print(f'Resampled X type {type(X_train)} y type {type(y_train_unencoded)}')
        print(f'Resampled X shape {X_train.shape} y shape {y_train_unencoded.shape}')

        # sampling converts series to nparray. Have to convert it back for the rest of the code
        y_train_unencoded = pd.DataFrame(y_train_unencoded)
        dist = y_train_unencoded.reset_index().groupby(0).count()
        dist.to_csv(f'{report_dir}/{sampler.__class__.__name__}-{len(labels)}-histogram.csv')

        print(f'Resampled distribution:\n{dist}')


    print(f'Shape of y_train {y_train_unencoded.shape}')
    print("One hot enocde label data...")

    categories = [sorted(labels.unique())]

    # have to convert back to array so it's not a sparse matrix
    y_train = OneHotEncoder(categories=categories).fit_transform(y_train_unencoded.values.reshape(-1, 1)).toarray()
    # y_train = OneHotEncoder(categories=categories).fit_transform(
    #     np.reshape(y_train_unencoded, (len(y_train_unencoded), 1))
    y_test = OneHotEncoder(categories=categories).fit_transform(y_test_unencoded.values.reshape(-1, 1)
                                                            ).toarray()


    return X_train, X_test, y_train, y_test, t


class ModelWrapper(object):

    report_file_name = None
    reports_dir = "reports"
    models_dir = "models"

    @staticmethod
    def set_report_filename(filename: str):
        ModelWrapper.report_file_name = filename

    @staticmethod
    def set_reports_dir(reports_dir: str):
        ModelWrapper.reports_dir = reports_dir

    @staticmethod
    def set_models_dir(models_dir: str):
        ModelWrapper.models_dir = models_dir


    @staticmethod
    def get_report_file_name(reports_dir: str, use_date=False):
        if ModelWrapper.report_file_name is not None:
            return f'{reports_dir}/{ModelWrapper.report_file_name}'
        if use_date:
            return  f"{reports_dir}/{datetime.now().strftime(DATE_FORMAT)}-dl_prototype-report.csv"
        return  f"{reports_dir}/dl_prototype-report.csv"


    @staticmethod
    def copy(mw):
        """
        Creates a deep copy of mw
        :param cls:
        :param mw:
        :return:
        """
        mw_copy = ModelWrapper(mw.model,
                               mw.model_name,
                               mw.architecture,
                               mw.feature_set_name,
                               mw.label_column,
                               mw.feature_column,
                               mw.data_file,
                               mw.sample_size_str,
                               mw.sampling_type,
                               mw.tokenizer,
                               mw.description,
                               mw.save_weights,
                               mw.optimizer_name,
                               mw.learning_rate,
                               mw.model_version,
                               mw.save_dir)

        mw_copy.tokenizer_file = mw.tokenizer_file
        mw_copy.train_time_min = mw.train_time_min
        mw_copy.test_predict_time_min = mw.predict_time_min
        mw_copy.evaluate_time_min = mw.evaluate_time_min
        mw_copy.train_evaluate_time_min = mw.train_evaluate_time_min
        mw_copy.network_history = mw.network_history
        mw_copy.weights_file = mw.weights_file
        mw_copy.epochs = mw.epochs
        mw_copy.X_train = mw.X_train
        mw_copy.X_test = mw.X_test
        mw_copy.y_train = mw.y_train
        mw_copy.y_test = mw.y_test
        mw_copy.test_scores = mw.scores
        mw_copy.train_scores = mw.train_scores
        mw_copy.test_confusion_matrix = mw.confusion_matrix
        mw_copy.test_roc_auc = mw.roc_auc
        mw_copy.test_fpr = mw.fpr
        mw_copy.test_tpr = mw.tpr
        mw_copy.test_crd = mw.crd
        mw_copy.learning_rate = mw.learnig_rate
        mw_copy.misc_items = mw.misc_items

        mw_copy.model_file = mw.model_file
        mw_copy.checkpoint_file = mw.checkpoint_file
        mw_copy.model_json_file = mw.model_json_file
        # mw_copy.network_history_file = mw.network_history_file
        mw_copy.weights_file = mw.weights_file
        mw_copy.report_file = mw.report_file
        mw_copy.tokenizer_file = mw.tokenizer_file
        mw_copy.saved_model_dir = mw.saved_model_dir

        return mw_copy

    def _load_model(self):
        """
        load model from file and set it in self.model

        :param load_model_file: path to model file
        :return:
        """
        if not os.path.exists(self.load_model_file):
            raise FileNotFoundError(f"{self.load_model_file} not found")

        print(f"Loaded model from: {self.load_model_file}")
        self.model = tf.keras.models.load_model(self.load_model_file)
        print(self.model.summary())

    def _set_output_files(self):
        basename = self._get_saved_file_basename()
        models_dir = f'{self.save_dir}/{ModelWrapper.models_dir}/{basename}'
        reports_dir = f'{self.save_dir}/{ModelWrapper.reports_dir}'
        log.debug(f"basename: {basename}")

        # model files
        self.model_file = f"{models_dir}/{basename}-model.h5"
        self.checkpoint_file = f"{models_dir}/checkpoints"
        self.model_json_file = f"{models_dir}/{basename}-model.json"
        self.weights_file = f"{models_dir}/{basename}-weights.h5"
        self.tokenizer_file = f'{models_dir}/{basename}-tokenizer.pkl'
        self.saved_model_dir = f"{models_dir}/{self.model_version}"

        self.report_file = ModelWrapper.get_report_file_name(reports_dir)


        if not os.path.exists(self.saved_model_dir):
            log.info(f'Creating {self.saved_model_dir}')
            os.makedirs(f'{self.saved_model_dir}', exist_ok = True)
        if not os.path.exists(reports_dir):
            log.info(f'Creating {reports_dir}')
            os.makedirs(f'{reports_dir}', exist_ok = True)


    def __init__(self,
                 model,
                 model_name,
                 architecture,
                 feature_set_name,
                 label_column,
                 feature_column,
                 data_file,
                 sample_size_str,
                 sampling_type="none",
                 tokenizer=None,
                 description=None,
                 save_weights=True,
                 optimizer_name = None,
                 learning_rate = None,
                 batch_size = 32,
                 model_version = 1,
                 save_dir = "drive/My Drive/Springboard/capstone",
                 load_model_file = None):
        """
        Constructor

        :param model:  keras model to train. Pass None and set load_model_file if you want to resume training
        :param model_name:  string name of model
        :param architecture:  architecture of model
        :param feature_set_name: feature set name
        :param label_column: name of column to use as label
        :param data_file: datafile used
        :param sample_size_str: string value summarizing sample size - ie, 1mil (required)
        :param sampling_type: specify type of sampling - ie, smote, nearmiss-2. default none
        :param tokenizer: tokenizer used to preprocess, default is None
        :param description: description of model. If not passed in, will automatically construct this
        :param save_weights: whether we should save weights for the model or not. Default is true to make old notebooks
            backwards compatible. However, for newer notebooks that use ModelCheckpoints,
            you should set this to false because ModelCheckpoint should already save the best model weights for you
        :param optimizer_name: name of optimizer to create -ie Adam
        :param learning_rate: learning rate for optimizer
        :param batch_size: batch size for training - default 32
        :param model_version: version for the model being trained - default 1
        :param save_dir: directory to save models and reports to - default drive/My Drive/Springboard/capstone
        :param load_model_file: if not None, wrapper will load the model from file instead of creating a new one
        """
        log.debug(f"Contructor ModelWrapper")

        self.model_name = model_name
        self.model = model
        self.feature_set_name = feature_set_name
        self.architecture = architecture
        self.label_column = label_column
        self.feature_column = feature_column
        self.data_file = data_file
        self.sample_size_str = sample_size_str
        self.sampling_type = sampling_type
        self.tokenizer = tokenizer
        self.description = description
        self.save_weights = save_weights
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.model_version = model_version
        self.save_dir = save_dir
        self.load_model_file = load_model_file


        self.tokenizer_file = None
        self.train_time_min = 0
        self.test_predict_time_min = 0
        self.evaluate_time_min = 0
        self.train_evaluate_time_min = 0
        self.network_history = None
        # number of epochs used to train
        self.epochs = 0
        # dumping ground for anything else we want to store
        self.misc_items = {}

        # set up variables to be used later
        self.X_test = None
        self.y_test = None

        self._set_output_files()

        if self.load_model_file is not None:
            # load model from path
            self._load_model()





        log.debug(f"Summary: {self}")




    def __str__(self):
        """
        Override __str__ so we can print a summary of this class
        
        :return: 
        """
        log.debug("ModelWrapper.__str__")
        summary = \
            f"\nModelWrapper parameters:\n" \
                f"\tmodel_name:\t\t\t{self.model_name}\n" \
                    f"\tdescription:\t\t\t{self.description}\n" \
                    f"\tarchitecture:\t\t\t{self.architecture}\n" \
                    f"\tfeature_set_name:\t\t{self.feature_set_name}\n" \
                    f"\tlabel_column:\t\t\t{self.label_column}\n" \
                    f"\tfeature_column:\t\t\t{self.feature_column}\n" \
                    f"\tdata_file:\t\t\t{self.data_file}\n" \
                    f"\tbatch_size:\t\t\t{self.batch_size}\n" \
                    f"\tsample_size:\t\t\t{self.sample_size_str}\n" \
                    f"\tsampling_type:\t\t\t{self.sampling_type}\n" \
                    f"\ttokenizer:\t\t\t{self.tokenizer}\n" \
                    f"\tsave_weights:\t\t\t{self.save_weights}\n" \
                    f"\toptimizer:\t\t\t{self.optimizer_name}\n" \
                    f"\tlearning_rate:\t\t\t{self.learning_rate}\n" \
                    f"\tversion:\t\t\t{self.model_version}\n" \
                    f"\tsave_dir:\t\t\t{self.save_dir}\n" \
                f"\tload_model_file:\t\t\t{self.load_model_file}\n" \
                f"\n\tModel Output:\n" \
                    f"\t\tmodel_file:\t\t\t{self.model_file}\n" \
                    f"\t\tcheckpoint_file:\t\t{self.checkpoint_file}\n" \
                    f"\t\tmodel_json_file:\t\t{self.model_json_file}\n" \
                    f"\t\tweights_file:\t\t\t{self.weights_file}\n" \
                    f"\t\tsaved_model_dir:\t\t{self.saved_model_dir}\n" \
                    f"\t\ttokenizer_file:\t\t\t{self.tokenizer_file}\n" \
                    f"\n\tReport Output:\n" \
                f"\t\treport_file:\t\t\t{self.report_file}\n"
                    # f"\t\tnetwork_history_file:\t\t{self.network_history_file}\n" \

        return summary

    def get_class_weight_dict(self, y_train):
        """
        computes class weight based on y_train distribution

        :param y_train: encoded labels - should have dimension (samples, num of classes)
        :return: dictionary of class weights ready to feed into model.fit function
        """
        self.y_train_unencoded = unencode(y_train)
        weights = compute_class_weight('balanced', np.arange(1, 6), self.y_train_unencoded)
        weights_dict = {i: weights[i] for i in np.arange(0, len(weights))}
        log.debug(f'class weights: {weights}')
        log.info(f'Computed class weight dictionary: {weights_dict}')
        return weights_dict


    # TODO: remove parameters defined in model.fit
    def fit(self, X_train, y_train,
            epochs,
            validation_split = 0.2,
            verbose = 1,
            callbacks = None,
            balance_class_weights = True,
            save_checkpoints = True):
        """
        Calls model.fit and record metrics

        Typical parameters for fit:
            epochs - max epochs for training
            batch_size
            validation_split
            verbose
            callbacks

        :param X_train:
        :param y_train:
        :param epochs:
        :param validation_split:
        :param verbose:
        :param callbacks:
        :param balance_class_weight:
        :return:
        """

        log.info(f'Number of training examples: {len(X_train)}')
        if balance_class_weights:
            self.class_weight = self.get_class_weight_dict(y_train)
        else:
            self.class_weight = None
        self.X_train = X_train
        self.y_train = y_train

        # add model checkpoint
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        # https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_options
        if save_checkpoints:
            log.info(f'Callbacks before adding checkpoints: {callbacks}')
            log.info(f"Adding {self.checkpoint_file} checkpoint callback...")
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath = self.checkpoint_file,
                verbose = 1,
                save_weights_only = False,
                monitor = 'val_loss',
               save_freq = 'epoch',
                save_best_only = True)

            callbacks.append(checkpoint)

        start_time = datetime.now()
        log.info(f'model: {self.model}')
        self.network_history = self.model.fit(X_train,
                                              y_train,
                                              batch_size=self.batch_size,
                                              epochs=epochs,
                                              validation_split=validation_split,
                                              callbacks=callbacks,
                                              verbose=verbose,
                                              class_weight=self.class_weight)
        end_time = datetime.now()
        self.train_time_min = round((end_time - start_time).total_seconds() / 60, 2)
        print(f'Total training time: {self.train_time_min} mins')
        self.epochs = len(self.network_history.history['loss'])
        return self.network_history

    def add(self, key, value):
        self.misc_items[key] = value

    def evaluate(self, X_test, y_test, verbose=1, unencoder=unencode):
        """
        Evalue our model

        :param X_test:
        :param y_test:
        :param verbose:
        :param unencoder: Optional - will use unencoder from this file if needed
        :return:
        """
        print(f'Number of test: {len(X_test)}')
        self.X_test = X_test
        self.y_test = y_test

        print("Running model.evaluate on test set...")
        start_time = datetime.now()
        self.test_scores = self.model.evaluate(X_test, y_test, verbose=verbose)
        end_time = datetime.now()
        self.evaluate_time_min = round((end_time - start_time).total_seconds() / 60, 2)

        print("Running model.predict on test set...")
        # this is a 2D array with each column as probabilyt of that class
        start_time = datetime.now()
        self.test_y_predict = self.model.predict(X_test)
        end_time = datetime.now()
        self.test_predict_time_min = round((end_time - start_time).total_seconds() / 60, 2)

        print("Unencode test set predictions...")
        # 1D array with class index as each value
        test_y_predict_unencoded = unencoder(self.test_y_predict)
        y_test_unencoded = unencoder(self.y_test)


        print("Generating test set confusion matrix...")
        self.test_confusion_matrix = confusion_matrix(y_test_unencoded, test_y_predict_unencoded)

        print("Calculating test set ROC AUC...")
        self.test_roc_auc, self.test_fpr, self.test_tpr = calculate_roc_auc(y_test, self.test_y_predict)

        print("Getting test set classification report...")
        self.test_classification_report = classification_report(y_test_unencoded, test_y_predict_unencoded)
        # classification report dictioonary
        self.test_crd = classification_report(y_test_unencoded, test_y_predict_unencoded, output_dict=True)

        """
        get stats from training set
        """

        print("Running model.evaluate on training set...")
        start_time = datetime.now()
        self.train_scores = self.model.evaluate(self.X_train, self.y_train, verbose=verbose)
        end_time = datetime.now()
        self.train_evaluate_time_min = round((end_time - start_time).total_seconds() / 60, 2)

        print("Running model.predict on training set...")
        # this is a 2D array with each column as probabilyt of that class
        start_time = datetime.now()
        self.train_y_predict = self.model.predict(self.X_train)
        end_time = datetime.now()
        self.train_predict_time_min = round((end_time - start_time).total_seconds() / 60, 2)

        print("Unencode training set predictions...")
        # 1D array with class index as each value
        train_y_predict_unencoded = unencoder(self.train_y_predict)


        print("Generating training set confusion matrix...")
        self.train_confusion_matrix = confusion_matrix(self.y_train_unencoded, train_y_predict_unencoded)

        print("Calculating training set ROC AUC...")
        self.train_roc_auc, self.train_fpr, self.train_tpr = calculate_roc_auc(self.y_train, self.train_y_predict)

        print("Getting training set classification report...")
        self.train_classification_report = classification_report(self.y_train_unencoded, train_y_predict_unencoded)
        # classification report dictioonary
        self.train_crd = classification_report(self.y_train_unencoded, train_y_predict_unencoded, output_dict=True)



    def _get_saved_file_basename(self):
        """
        contructs prefixes for all of our saved model and report files
        :return: str: prefix of file
        """
        if self.sample_size_str is not None:
            description = f"{self.model_name}-" \
                    f"{self.architecture}-" \
                    f"{self.feature_set_name}-" \
                    f"sampling_{self.sampling_type}-" \
                    f"{self.sample_size_str}-" \
                    f"{self.feature_column}-" \
                    f"v{self.model_version}"
        else:
            description = f"{self.model_name}-" \
                    f"{self.architecture}-" \
                    f"{self.feature_set_name}-" \
                    f"sampling_{self.sampling_type}-" \
                    f"{self.feature_column}-" \
                    f"v{self.model_version}"

        log.info(f"saved file basename: {description}")
        return description


    def save(self, save_format=None, append_report=True):
        """
        Save the following information based on our trained model:
        * model file
        * model weights (if save_weights is True)
        * writes model report
        * tokenizer used for pre-processing

        :param save_dir: base directory to save files models and reports will be appended to this. will create reports and models dir underneath. this is depreciated and will be removed later
        :param save_format: save_format for tf.model.save. Default None
        :param append_report: if existing report, True to append or False to overwrite. Default True
        :return:
        """
        self._set_output_files()

        print(f"Saving to report file: {self.report_file}")
        report = self.get_report()
        report.save(self.report_file, append=append_report)

        print(f"Saving json config file: {self.model_json_file}")
        if self.model:
            model_json = self.model.to_json()
            with open(self.model_json_file, 'w') as json_file:
                json_file.write(model_json)

        print(f"Saving weights file: {self.weights_file}")
        if self.weights_file is not None and self.save_weights:
            self.model.save_weights(self.weights_file,
                                    save_format=save_format)

        # if self.network_history is not None:
        #     print(f"Saving history file: {self.network_history_file}")
        #     with open(self.network_history_file, 'w') as file:
        #         json.dump(self.network_history.history,
        #                     file,
        #                     default=to_serializable)

        print(f"Saving model file: {self.model_file}")
        self.model.save(self.model_file, save_format=save_format)

        print(f"Saving SavedModel to: {self.saved_model_dir}")
        tf.saved_model.save(self.model, self.saved_model_dir)

        if self.tokenizer is not None:
            log.info(f"Saving tokenizer file: {self.tokenizer_file}")
            pickle.dump(self.tokenizer, open(self.tokenizer_file, 'wb'))




    def get_report(self):
        if self.description is not None:
            report = ModelReport(self.model_name, self.architecture, self.description)
        else:
            report = ModelReport(self.model_name, self.architecture, self._get_saved_file_basename())

        report.add("feature_column", self.test_crd)
        report.add("label_column", self.test_crd)
        report.add("classification_report", self.test_crd)
        report.add("classification_report_train", self.train_crd)
        report.add("roc_auc", self.test_roc_auc)
        report.add("roc_auc_train", self.train_roc_auc)
        report.add("loss", self.test_scores[0])
        report.add("accuracy", self.test_scores[1])
        report.add("loss_train", self.train_scores[0])
        report.add("accuracy_train", self.train_scores[1])
        report.add("confusion_matrix", json.dumps(self.test_confusion_matrix.tolist()))
        report.add("confusion_matrix_train", json.dumps(self.train_confusion_matrix.tolist()))
        report.add("file", self.data_file)
        # too long to save in CSV
        report.add("optimizer", self.optimizer_name)
        report.add("learning_rate", self.learning_rate)
        report.add("version", self.model_version)
        report.add("save_dir", self.save_dir)
        report.add("network_history", self.network_history.history)
        report.add("tokenizer_file", self.tokenizer_file)
        report.add("batch_size", self.batch_size)
        report.add("epochs", self.epochs)
        report.add("feature_set_name", self.feature_set_name)
        if self.class_weight is not None:
            report.add("class_weight", self.class_weight)
        report.add("sample_size_str", self.sample_size_str)
        report.add("sampling_type", self.sampling_type)
        report.add("model_file", self.model_file)
        report.add("checkpoint_dir", self.checkpoint_file)
        report.add("model_json_file", self.model_json_file)
        report.add("weights_file", self.weights_file)
        report.add("saved_model_dir", self.saved_model_dir)
        report.add("test_examples", self.X_test.shape[0])
        report.add("test_features", self.X_test.shape[1])
        report.add("train_examples", self.X_train.shape[0])
        report.add("train_features", self.X_train.shape[1])
        report.add("train_time_min", self.train_time_min)
        report.add("evaluate_time_min", self.evaluate_time_min)
        report.add("evaluate_time_min_train", self.train_evaluate_time_min)
        report.add("predict_time_min", self.test_predict_time_min)
        report.add("predict_time_min_train", self.train_predict_time_min)
        report.add("status", "success")
        report.add("status_date", datetime.now().strftime(TIME_FORMAT))
        for k, v in self.misc_items.items():
            report.add(k, v)

        return report

class EmbeddingModelWrapper(ModelWrapper, ABC):
    """
    Abstract base class for creating models that use an initial embedding layer

    Must implement the following abstract functions:
        build_model

    Must override the following functions:
        __init__
        __str__
    """

    # TODO: implement copy static method

    @abstractmethod
    def build_model(self):
        """
        implement this method to create model based on class variables
        :return: model object used for training
        """
        return NotImplemented

    def __init__(self,
                 vocab_size,
                 max_sequence_length = 100,
                 embed_size = 300,
                 train_embeddings = False,
                 *args,
                 **kwargs):
        """

        :param vocab_size: size of corpus vocabulary + 1
        :param max_sequence_length: max length of padded sequence. default 100
        :param embed_size: vector length of embedding. default 300
        :param train_embeddings: if set to True then the first embedding layer is not trainable. default False
        :param args:
        :param kwargs:
        """
        log.debug(f'Constructor EmbeddingModelWrapper')

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_sequence_length = max_sequence_length
        self.train_embeddings = train_embeddings

        # pass None to super constructor since we haven't built it yet
        super().__init__(None, *args, **kwargs)

        if self.load_model_file is None:
            self.model = self.build_model()



    def __str__(self):
        log.debug("EmbeddingModelWrapper.__str__")
        super_sum = super().__str__()
        summary = f"{super_sum}\n" \
            f"\nEmbeddingModelWrapper parameters:\n" \
            f"\tvocab_size:\t\t\t{self.vocab_size}\n" \
            f"\tembed_size:\t\t\t{self.embed_size}\n" \
            f"\tmax_sequence_length:\t\t{self.max_sequence_length}\n" \
            f"\ttrain_embeddings:\t\t{self.train_embeddings}\n"
        return summary

    def get_report(self):
        report = super().get_report()
        report.add("embedding", self.embed_size)
        report.add("vocab_size", self.vocab_size)
        if self.X_train is not None:
            report.add("max_sequence_length", self.X_train.shape[1])
        report.add("train_embeddings", self.train_embeddings)

        return report



class LSTM1LayerModelWrapper(EmbeddingModelWrapper):


    # TODO: implement copy static method

    def __init__(self,
                 lstm_dim,
                 dropout_rate,
                 recurrent_dropout_rate,
                 bidirectional = False,
                 *args,
                 **kwargs):
        log.debug(f'Constructor LSTM1LayerModelWrapper')

        self.lstm_dim = lstm_dim
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate

        super().__init__(*args, **kwargs)


    def build_model(self):
        """
        Build a 1 layer LSTM network
        :return: LSTM model
        """
        log.debug(f'Building Model: {self}')

        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size,
                            output_dim=self.embed_size,
                            input_length=self.max_sequence_length,
                            trainable = self.train_embeddings))
        if self.bidirectional:
            model.add(Bidirectional(LSTM(self.lstm_dim,
                                         dropout = self.dropout_rate,
                                         recurrent_dropout = self.recurrent_dropout_rate)))
        else:
            model.add(LSTM(self.lstm_dim,
                           dropout = self.dropout_rate,
                           recurrent_dropout = self.recurrent_dropout_rate))
        model.add(Dense(5, activation="softmax"))

        model.compile(loss="categorical_crossentropy",
                      optimizer=eval(self.optimizer_name)(learning_rate = self.learning_rate),
                      metrics=["categorical_accuracy"])

        print(f"Build model:\n{model.summary()}")
        return model

    def _get_saved_file_basename(self) -> str:
        """
        Override default descripition to take into account hyperparameters
        :return:
        """
        if self.sample_size_str is not None:
            description = f"{self.model_name}-" \
                    f"{self.architecture}-" \
                    f"dr{get_decimal_str(self.dropout_rate)}-" \
                    f"rdr{get_decimal_str(self.recurrent_dropout_rate)}-" \
                    f"batch{self.batch_size}-" \
                    f"lr{get_decimal_str(self.learning_rate)}-" \
                    f"{self.feature_set_name}-" \
                    f"sampling_{self.sampling_type}-" \
                    f"{self.sample_size_str}-" \
                    f"{self.feature_column}-" \
                    f"v{self.model_version}"
        else:
            description = f"{self.model_name}-" \
                f"{self.architecture}-" \
                f"dr{get_decimal_str(self.dropout_rate)}-" \
                f"rdr{get_decimal_str(self.recurrent_dropout_rate)}-" \
                f"batch{self.batch_size}-" \
                f"lr{get_decimal_str(self.learning_rate)}-" \
                f"{self.feature_set_name}-" \
                f"sampling_{self.sampling_type}-" \
                f"{self.feature_column}" \
                f"v{self.model_version}"
        return description

    def __str__(self):
        log.debug("LSTM1LayerModelWrapper.__str__")
        super_sum = super().__str__()
        summary = f"{super_sum}\n" \
            f"LSTM1LayerModelWrapper parameters:\n" \
            f"\tlstm_dim:\t\t\t{self.lstm_dim}\n" \
            f"\tbidirectional:\t\t\t{self.bidirectional}\n" \
            f"\tdropout_rate:\t\t\t{self.dropout_rate}\n" \
            f"\trecurrent_dropout_rate:\t\t{self.recurrent_dropout_rate}\n"
        return summary

    def get_report(self):
        report = super().get_report()
        report.add("lstm_dim", self.lstm_dim)
        report.add("bidirectional", self.bidirectional)
        report.add("dropout_rate", self.dropout_rate)
        report.add("recurrent_dropout_rate", self.recurrent_dropout_rate)

        return report



# define functions to be used for json serialization for float32
# is was causing and error when converting network history to json format
#
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'np' is not defined
# >>> import numpy as np
# >>> @to_serializable.register(np.float32)
# ... def ts_float32(val):
# ...     """Used if *val* is an instance of numpy.float32."""
# ...     return np.float64(val)
# Solution: https://ellisvalentiner.com/post/serializing-numpyfloat32-json/
@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


class ModelReport(object):
    """
    dict wrapper that represents a report from model execution
    """

    # these columns will be json objects
    json_columns = ["classification_report", "fpr", "tpr", "confusion_matrix"]

    @staticmethod
    def load_report(filename: str, index: int):
        """
        Uses a specific row in a report file and re-create the ModelReport
        :param filename:
        :param index:
        :return:
        """
        assert os.path.exists(filename), f'File not found: {filename}'
        df = pd.read_csv(filename, quotechar=",").iloc[index]
        report = ModelReport(df.model_name, df.architecture, df.description)

        df = df.drop(["model_name", "architecture", "description"])
        for col, value in df.items():
            report.add(col, df[col])
        return report


    def __init__(self, model_name, architecture: str, description=None):
        self.report = {}
        self.report["model_name"] = model_name
        self.report["architecture"] = architecture
        self.report["description"] = description

    def add(self, key, value = None):
        """
        add value to report using key
        :param key:
        :param value:
        :return:
        """
        log.debug(f"\tadding to report: key: {key} value: {value}")
        # log.debug(f"key type: {type(key)} value type: {type(value)}")
        if value is not None:
            # if it's a list then we serialize to json string format so we can load it back later
            if isinstance(value, list):
                log.debug(f"converting to list to json: {value}")
                log.debug(f'json dumps results {json.dumps(value)}')
                self.report[key] = json.dumps(value)
                log.debug("done converting list to json")
            elif isinstance(value, dict):
                log.debug(f"converting to dict to json: {value}")
                dict_converted = {str(k): value[k] for k in value.keys()}
                log.debug(f'json dumps results {json.dumps(dict_converted, default=to_serializable)}')
                self.report[key] = json.dumps(dict_converted, default=to_serializable)
                log.debug("done converting dict to json")
            elif isinstance(value, np.ndarray):
                log.debug(f"converting ndarray to json: {value}")
                log.debug(f'json dumps results {json.dumps(value)}')
                self.report[key] = json.dumps(value.tolist())
                log.debug("done converting ndarray to json")
            else:
                self.report[key] = value

    def to_df(self):
        log.debug(self.report)
        df = pd.DataFrame()
        return df.append(self.report, ignore_index=True, sort=False)

    def save(self, report_file, append=True):
        """
        Saves a report to CSV format

        :param self:
        :param report_file: filepath of report
        :param append: append to existing report. default True
        :return:
        """
        # check to see if report file exisits, if so load it and append
        exists = os.path.isfile(report_file)
        if append and exists:
            print(f'Loading to append to: {report_file}')
            report_df = pd.read_csv(report_file, quotechar="'")
        else:
            report_df = pd.DataFrame()

        report_df = report_df.append(self.to_df(), ignore_index=True, sort=False)
        print("Saving report file...")
        report_df.to_csv(report_file, index=False, quotechar="'")
        return report_df

    def get(self, key):
        """
        Get object or value from report
        :param key: string key
        :return: None if not in report, else value
        """
        ret = None
        if key in self.report.keys():
            ret = self.report[key]
        else:
            log.warn(f'Key not found in report: {key}')
        return ret


    def __str__(self):
        return str(self.to_df())


# define our attention layer for later
class AttentionLayer(Layer):

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        """

        self.supports_masking = True
        self.init = K.initializers.get('glorot_uniform')

        self.W_regularizer = K.regularizers.get(W_regularizer)
        self.b_regularizer = K.regularizers.get(b_regularizer)

        self.W_constraint = K.constraints.get(W_constraint)
        self.b_constraint = K.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # TF backend doesn't support it
        # eij = K.dot(x, self.W)
        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = {'step_dim': self.step_dim}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
