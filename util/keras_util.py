from sklearn.metrics import roc_curve, auc
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
import sys


DATE_FORMAT = '%Y-%m'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

log = logging.getLogger()



def calculate_roc_auc(y_test: np.ndarray, y_predict: pd.DataFrame):
    """
    Caculates the false positive rate, true positive rate, and roc_auc
    Return:
      roc_auc - dictinary of AUC values there will be 1 auc per class then a macro
          and micro for the overall model
          keys: auc_0 to auc_{n_classes} + micro and macro
      fpr - dictionary of FPR, each value will be a n_classes x ? interpolated array
          so you can plt ROC curve for each class
          keys: 0 to {n_classes} + micro and macro
      tpr - dictionary of TPR, each value will be a n_classes x ? interpolated array
          so used to plt ROC curve for each class
          keys: 0 to {n_classes} + micro and macro

    :param y_test: test labels. should be np array or dataframe
    :param y_predict: test predictions - this should either be an np.ndarray or DataFrame
    :return:
    """
    if isinstance(y_predict, np.ndarray):
        y_predict =  pd.DataFrame(y_predict)


    n_classes = y_test.shape[1]

    # to compute the micro RUC/AUC, we need binariized labels (y_test) and probability of predictions y_predict_df

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in np.arange(0, 5):
        fpr_ndarray, tpr_ndarray, _ = roc_curve(y_test[:, i], y_predict[i].to_list())
        fpr[str(i)] = fpr_ndarray.tolist()
        tpr[str(i)] = tpr_ndarray.tolist()
        roc_auc[f'auc_{i + 1}'] = auc(fpr[str(i)], tpr[str(i)]).tolist()

    # Compute micro-average ROC curve and ROC area
    fpr_ndarray, tpr_ndarray, _ = roc_curve(y_test.ravel(), y_predict.values.ravel())
    fpr["micro"] = fpr_ndarray.tolist()
    tpr["micro"] = tpr_ndarray.tolist()
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"]).tolist()

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[str(i)] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[str(i)], tpr[str(i)])

    # Finally aveage it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr.tolist()
    tpr["macro"] = mean_tpr.tolist()
    roc_auc["auc_macro"] = auc(fpr["macro"], tpr["macro"]).tolist()

    return roc_auc, fpr, tpr


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


def preprocess_file(data_df, feature_column, label_column, max_sequence_length = 100, use_oov_token=True):
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
    :return: X_train, X_test, y_train, y_test, tokenizer
    """
    labels = data_df[label_column]
    features = data_df[feature_column]

    print("One hot enocde label data...")
    y = OneHotEncoder().fit_transform(labels.values.reshape(len(labels), 1)).toarray()

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
    X_train, X_test, y_train, y_test = train_test_split(features_padded, y, random_state=1)
    print(f'Training X shape {X_train.shape} y shape {y_train.shape}')
    print(f'Test X shape {X_test.shape} y shape {y_test.shape}')

    return X_train, X_test, y_train, y_test, t


class ModelWrapper(object):

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
                               mw.description)
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
                 data_file,
                 sampling_type="none",
                 embed_size = None,
                 tokenizer=None,
                 description=None):
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
        """
        self.model_name = model_name
        self.model = model
        self.feature_set_name = feature_set_name
        self.architecture = architecture
        self.label_column = label_column
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
        # dumping ground for anything else we want to store
        self.misc_items = {}


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
        self.batch_size = 128
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
            description = f"{self.model_name}-{self.architecture}-{self.feature_set_name}-sampling_{self.sampling_type}-{self.X_test.shape[0] + self.X_train.shape[0]}-{self.X_test.shape[1]}-{self.label_column}"
        else:
            description = self.model_name
        return description

    @staticmethod
    def get_report_file_name(save_dir: str, use_date=True):
        if use_date:
            return  f"{save_dir}/reports/{datetime.now().strftime(DATE_FORMAT)}-dl_prototype-report.csv"
        return  f"{save_dir}/reports/dl_prototype-report.csv"

    def save(self, save_dir, save_format=None, append_report=True):
        description = self._get_description()
        print(f"description: {description}")

        self.model_file = f"{save_dir}/models/{description}-model.h5"
        self.model_json_file = f"{save_dir}/models/{description}-model.json"
        self.weights_file = f"{save_dir}/models/{description}-weights.h5"
        self.report_file = ModelWrapper.get_report_file_name(save_dir)
        self.tokenizer_file = f'{save_dir}/models/dl-tokenizer.pkl'

        print(f"Saving model file: {self.model_file}")
        self.model.save(self.model_file, save_format=save_format)

        print(f"Saving json config file: {self.model_json_file}")
        if self.model:
            model_json = self.model.to_json()
            with open(self.model_json_file, 'w') as json_file:
                json_file.write(model_json)

        print(f"Saving weights file: {self.weights_file}")
        if self.weights_file is not None:
            self.model.save_weights(self.weights_file,
                    save_format=save_format)


        if self.tokenizer is not None:
            log.info(f"Saving tokenizer file: {self.tokenizer_file}")
            pickle.dump(self.tokenizer, open(self.tokenizer_file, 'wb'))

        print(f"Saving to report file: {self.report_file}")
        report = self.get_report()
        report.save(self.report_file, append=append_report)


    def get_report(self):
        if self.description is not None:
            report = ModelReport(self.model_name, self.architecture, self.description)
        else:
            report = ModelReport(self.model_name, self.architecture, self._get_description())

        report.add("classification_report", self.crd)
        report.add("roc_auc", self.roc_auc)
        report.add("loss", self.scores[0])
        report.add("accuracy", self.scores[1])
        report.add("confusion_matrix", json.dumps(self.confusion_matrix.tolist()))
        report.add("file", self.data_file)
        # too long to save in CSV
        # report.add("network_history_file", self.network_history_file)
        # report.add("history", self.network_history.history)
        report.add("tokenizer_file", self.tokenizer_file)
        if self.X_train is not None:
            report.add("max_sequence_length", self.X_train.shape[1])
        report.add("batch_size", self.batch_size)
        report.add("epochs", self.epochs)
        report.add("feature_set_name", self.feature_set_name)
        if self.class_weight is not None:
            report.add("class_weights", self.class_weight)
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
            if isinstance(value, list) or isinstance(value, dict):
                log.debug("converting to json")
                self.report[key] = json.dumps(value)
            elif isinstance(value, np.ndarray):
                log.debug("converting to json")
                self.report[key] = json.dumps(value.tolist())
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
