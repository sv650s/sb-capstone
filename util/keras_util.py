from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Layer
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import util.file_util as fu
import json
import logging
import os
from datetime import datetime
import pickle

from util.metric_util import calculate_roc_auc

DATE_FORMAT = '%Y-%m'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

log = logging.getLogger()


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
    log.debug(f'Unencoded: {ret}')
    return ret


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

    @staticmethod
    def set_report_filename(filename: str):
        ModelWrapper.report_file_name = filename


    @staticmethod
    def get_report_file_name(save_dir: str, use_date=True):
        if ModelWrapper.report_file_name is not None:
            return f'{save_dir}/reports/{ModelWrapper.report_file_name}'
        if use_date:
            return  f"{save_dir}/reports/{datetime.now().strftime(DATE_FORMAT)}-dl_prototype-report.csv"
        return  f"{save_dir}/reports/dl_prototype-report.csv"


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
                               mw.data_file,
                               mw.sampling_type,
                               mw.embed_size,
                               mw.tokenizer,
                               mw.description,
                               mw.feature_column)
        mw_copy.tokenizer_file = mw.tokenizer_file
        mw_copy.train_time_min = mw.train_time_min
        mw_copy.predict_time_min = mw.predict_time_min
        mw_copy.evaluate_time_min = mw.evaluate_time_min
        mw_copy.network_history = mw.network_history
        mw_copy.weights_file = mw.weights_file
        mw_copy.X_train = mw.X_train
        mw_copy.X_test = mw.X_test
        mw_copy.y_train = mw.y_train
        mw_copy.y_test = mw.y_test
        mw_copy.scores = mw.scores
        mw_copy.confusion_matrix = mw.confusion_matrix
        mw_copy.roc_auc = mw.roc_auc
        mw_copy.fpr = mw.fpr
        mw_copy.tpr = mw.tpr
        mw_copy.crd = mw.crd
        return mw_copy


    def __init__(self,
                 model,
                 model_name,
                 architecture,
                 feature_set_name,
                 label_column,
                 feature_column,
                 data_file,
                 sampling_type="none",
                 embed_size = None,
                 tokenizer=None,
                 description=None,
                 save_weights=True):
        """
        Constructor

        :param model:  keras model
        :param model_name:  string name of model
        :param architecture:  architecture of model
        :param feature_set_name: feature set name
        :param label_column: name of column to use as label
        :param data_file: datafile used
        :param sampling_type: specify type of sampling - ie, smote, nearmiss-2. default none
        :param embed_size: size of embedding. default is None
        :param tokenizer: tokenizer used to preprocess, default is None
        :param description: description of model. If not passed in, will automatically construct this
        :param save_weights: whether we should save weights for the model or not. Default is true to make old notebooks
            backwards compatible. However, for newer notebooks that use ModelCheckpoints,
            you should set this to false because ModelCheckpoint should already save the best model weights for you
        """
        self.model_name = model_name
        self.model = model
        self.feature_set_name = feature_set_name
        self.architecture = architecture
        self.label_column = label_column
        self.feature_column = feature_column
        self.data_file = data_file
        self.batch_size = 0
        self.sampling_type = sampling_type
        self.embed_size = embed_size
        self.tokenizer = tokenizer
        self.description = description
        self.tokenizer_file = None
        self.train_time_min = 0
        self.predict_time_min = 0
        self.evaluate_time_min = 0
        self.network_history = None
        self.weights_file = None
        self.epochs = 0
        self.save_weights = save_weights
        # dumping ground for anything else we want to store
        self.misc_items = {}

        self.X_test = None
        self.y_test = None


    def fit(self, X_train, y_train,
            epochs,
            batch_size = 32,
            validation_split=0.2,
            verbose=1,
            callbacks=None,
            class_weight = None):
        print(f'Number of training examples: {len(X_train)}')
        start_time = datetime.now()
        self.network_history = self.model.fit(X_train,
                                              y_train,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              validation_split=validation_split,
                                              callbacks=callbacks,
                                              verbose=verbose,
                                              class_weight=class_weight)
        end_time = datetime.now()
        self.class_weight = class_weight
        self.train_time_min = round((end_time - start_time).total_seconds() / 50, 2)
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
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

        print("Running model.evaluate...")
        start_time = datetime.now()
        self.scores = self.model.evaluate(X_test, y_test, verbose=verbose)
        end_time = datetime.now()
        self.evaluate_time_min = round((end_time - start_time).total_seconds() / 50, 2)

        print("Running model.predict...")
        # this is a 2D array with each column as probabilyt of that class
        start_time = datetime.now()
        self.y_predict = self.model.predict(X_test)
        end_time = datetime.now()
        self.predict_time_min = round((end_time - start_time).total_seconds() / 50, 2)

        print("Unencode predictions...")
        # 1D array with class index as each value
        y_predict_unencoded = unencoder(self.y_predict)
        y_test_unencoded = unencoder(self.y_test)


        print("Generating confusion matrix...")
        self.confusion_matrix = confusion_matrix(y_test_unencoded, y_predict_unencoded)

        print("Calculating ROC AUC...")
        self.roc_auc, self.fpr, self.tpr = calculate_roc_auc(y_test, self.y_predict)

        print("Getting classification report...")
        self.classification_report = classification_report(y_test_unencoded, y_predict_unencoded)
        # classification report dictioonary
        self.crd = classification_report(y_test_unencoded, y_predict_unencoded, output_dict=True)

    def _get_description(self):
        directory, inbasename = fu.get_dir_basename(self.data_file)
        if self.X_test is not None:
            # self.X_test might not be set yet
            description = f"{self.model_name}-{self.architecture}-{self.feature_set_name}-sampling_{self.sampling_type}-{self.X_test.shape[0] + self.X_train.shape[0]}-{self.X_test.shape[1]}-{self.feature_column}"
        else:
            description = self.model_name
        return description

    def get_weights_filename(self, save_dir):
        """
        Returns the name of the weights file for this model.

        Mean to be used when we use ModelCheckpoint so we can tell ModelCheckpoint
        where to save the weights file.

        :param save_dir:
        :return:
        """
        return f"{save_dir}/models/{self._get_description()}-weights.h5"

    def save(self, save_dir, save_format=None, append_report=True):
        """
        Save the following information based on our trained model:
        * model file
        * model weights (if save_weights is True)
        * writes model report
        * tokenizer used for pre-processing

        :param save_dir: base directory to save files models and reports will be appended to this
        :param save_format: save_format for tf.model.save
        :param append_report: if existing report, True to append or False to overwrite
        :return:
        """
        description = self._get_description()
        print(f"description: {description}")

        self.model_file = f"{save_dir}/models/{description}-model.h5"
        self.model_json_file = f"{save_dir}/models/{description}-model.json"
        self.network_history_file = f"{save_dir}/reports/{description}-history.pkl"
        self.weights_file = self.get_weights_filename(save_dir)
        self.report_file = ModelWrapper.get_report_file_name(save_dir)
        self.tokenizer_file = f'{save_dir}/models/{description}-tokenizer.pkl'

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

        # TODO: should probably save this as json instead
        if self.network_history is not None:
            print(f"Saving history file: {self.network_history_file}")
            with open(self.network_history_file, 'wb') as file:
                pickle.dump(self.network_history.history, file)

        print(f"Saving model file: {self.model_file}")
        self.model.save(self.model_file, save_format=save_format)

        if self.tokenizer is not None:
            log.info(f"Saving tokenizer file: {self.tokenizer_file}")
            pickle.dump(self.tokenizer, open(self.tokenizer_file, 'wb'))




    def get_report(self):
        if self.description is not None:
            report = ModelReport(self.model_name, self.architecture, self.description)
        else:
            report = ModelReport(self.model_name, self.architecture, self._get_description())

        report.add("feature_column", self.crd)
        report.add("label_column", self.crd)
        report.add("classification_report", self.crd)
        report.add("roc_auc", self.roc_auc)
        report.add("loss", self.scores[0])
        report.add("accuracy", self.scores[1])
        report.add("confusion_matrix", json.dumps(self.confusion_matrix.tolist()))
        report.add("file", self.data_file)
        # too long to save in CSV
        report.add("network_history_file", self.network_history_file)
        # report.add("history", self.network_history.history)
        report.add("tokenizer_file", self.tokenizer_file)
        if self.X_train is not None:
            report.add("max_sequence_length", self.X_train.shape[1])
        report.add("batch_size", self.batch_size)
        report.add("epochs", self.epochs)
        report.add("feature_set_name", self.feature_set_name)
        if self.class_weight is not None:
            report.add("class_weight", self.class_weight)
        report.add("sampling_type", self.sampling_type)
        report.add("embedding", self.embed_size)
        report.add("model_file", self.model_file)
        report.add("model_json_file", self.model_json_file)
        report.add("weights_file", self.weights_file)
        report.add("test_examples", self.X_test.shape[0])
        report.add("test_features", self.X_test.shape[1])
        report.add("train_examples", self.X_train.shape[0])
        report.add("train_features", self.X_train.shape[1])
        report.add("train_time_min", self.train_time_min)
        report.add("evaluate_time_min", self.evaluate_time_min)
        report.add("predict_time_min", self.predict_time_min)
        report.add("status", "success")
        report.add("status_date", datetime.now().strftime(TIME_FORMAT))
        for k, v in self.misc_items.items():
            report.add(k, v)

        return report




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
                log.debug(f'json dumps results {json.dumps(dict_converted)}')
                self.report[key] = json.dumps(dict_converted)
                log.debug("done converting dict to json")
            elif isinstance(value, np.ndarray):
                log.debug(f"converting ndarray to json: {value}")
                log.debug(f'json dumps results {json.dumps(value)}')
                self.report[key] = json.dumps(value.tolist())
                log.debug("done converting ndarray to json")
            else:
                self.report[key] = value

    def to_df(self):
        print(self.report)
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
