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


DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

log = logging.getLogger()



def calculate_roc_auc(y_test: np.ndarray, y_score: pd.DataFrame):
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

    :param y_test: test labels
    :param y_score: test predictions - this should either be an np.ndarray or DataFrame
    :return:
    """
    if isinstance(y_score, np.ndarray):
        y_score =  pd.DataFrame(y_score)


    n_classes = y_test.shape[1]

    # to compute the micro RUC/AUC, we need binariized labels (y_test) and probability of predictions y_predict_df

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in np.arange(0, 5):
        fpr_ndarray, tpr_ndarray, _ = roc_curve(y_test[:, i], y_score[i].to_list())
        fpr[str(i)] = fpr_ndarray.tolist()
        tpr[str(i)] = tpr_ndarray.tolist()
        roc_auc[f'auc_{i + 1}'] = auc(fpr[str(i)], tpr[str(i)]).tolist()

    # Compute micro-average ROC curve and ROC area
    fpr_ndarray, tpr_ndarray, _ = roc_curve(y_test.ravel(), y_score.values.ravel())
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

    return [row.idxmax() + 1 for index, row in input_df.iterrows()]


def preprocess_file(data_df, feature_column, label_column, keep_percentile, use_oov_token=True):
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
    :param keep_percentile: percentile of feature length to keep - all features will be padded to this length
    :param use_oov_token: Default True. Use a out of vocabulary token for tokenizer
    :return:
    """
    labels = data_df[label_column]
    features = data_df[feature_column]

    print("One hot enocde label data...")
    y = OneHotEncoder().fit_transform(labels.values.reshape(len(labels), 1)).toarray()

    # split our data into train and test sets
    print("Splitting data into training and test sets...")
    features_train, features_test, y_train, y_test = train_test_split(features, y, random_state=1)

    # Pre-process our features (review body)
    if use_oov_token:
        t = Tokenizer(lower=True, oov_token="<UNK>")
    else:
        t = Tokenizer(lower=True)

    # fit the tokenizer on the documents
    t.fit_on_texts(features_train)
    # tokenize both our training and test data
    train_sequences = t.texts_to_sequences(features_train)
    test_sequences = t.texts_to_sequences(features_test)

    print("Vocabulary size={}".format(len(t.word_index)))
    print("Number of Documents={}".format(t.document_count))

    # figure out 99% percentile for our max sequence length
    data_df["feature_length"] = data_df.review_body.apply(lambda x: len(x.split()))
    max_sequence_length = int(data_df.feature_length.quantile([keep_percentile]).values[0])
    print(f'Max Sequence Length: {max_sequence_length}')

    # pad our reviews to the max sequence length
    X_train = sequence.pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test = sequence.pad_sequences(test_sequences, maxlen=max_sequence_length)

    return X_train, X_test, y_train, y_test, t, max_sequence_length


class ModelWrapper(object):

    def __init__(self, model, name, label_name, data_file, embedding, tokenizer=None, description=None):
        self.name = name
        self.model = model
        self.label_name = label_name
        self.data_file = data_file
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.description = description
        # dumping ground for anything else we want to store
        self.misc_items = {}


    def fit(self, X_train, y_train, batch_size, epochs, validation_split=0.2, verbose=1, callbacks=None):
        start_time = datetime.now()
        self.network_history = self.model.fit(X_train,
                                              y_train,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              validation_split=validation_split,
                                              callbacks=callbacks,
                                              verbose=verbose)
        end_time = datetime.now()
        self.train_time_min = round((end_time - start_time).total_seconds() / 50, 2)
        self.X_train = X_train
        self.y_train = y_train
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
            description = f"{inbasename}-{self.name}-{self.X_test.shape[1]}-{self.label_name}"
        else:
            description = self.name
        return description


    def save(self, save_dir, save_history=True, save_format=None, append_report=True):
        description = self._get_description()
        print(f"description: {description}")

        self.model_file = f"{save_dir}/models/{description}-model.h5"
        self.model_json_file = f"{save_dir}/models/{description}-model.json"
        self.weights_file = f"{save_dir}/models/{description}-weights.h5"
        self.network_history_file = f'{save_dir}/models/{description}-history.pkl'
        self.report_file = f"{save_dir}/reports/{datetime.now().strftime(DATE_FORMAT)}-dl_prototype-report.csv"
        self.tokenizer_file = f'{save_dir}/models/dl-tokenizer.pkl'

        print(f"Saving model file: {self.model_file}")
        try:
            self.model.save(self.model_file,
                    save_format=save_format)
        except Exception as e:
            print("Unexpected error:", sys.exc_info()[0])
            self.model_file = None

        print(f"Saving json config file: {self.model_json_file}")
        try:
            model_json = self.model.to_json()
            with open(self.model_json_file, 'w') as json_file:
                json_file.write(model_json)
        except Exception as e:
            print("Unexpected error:", sys.exc_info()[0])
            self.model_json_file = None

        print(f"Saving weights file: {self.weights_file}")
        try:
            self.model.save_weights(self.weights_file,
                    save_format=save_format)
        except Exception as e:
            print("Unexpected error:", sys.exc_info()[0])
            self.weights_file = None


        print(f"Saving network history file: {self.network_history_file}")
        try:
            pickle.dump(self.network_history, open(self.network_history_file, 'wb'))
        except Exception as e:
            print("Unexpected error:", sys.exc_info()[0])
            self.network_history_file = None

        if self.tokenizer:
            log.info(f"Saving tokenizer file: {self.tokenizer_file}")
            try:
                pickle.dump(self.tokenizer, open(self.tokenizer_file, 'wb'))
            except Exception as e:
                print("Unexpected error:", sys.exc_info()[0])
                self.tokenizer_file = None

        print(f"Saving to report file: {self.report_file}")
        report = self.get_report()
        report.save(self.report_file, append=append_report)


    def get_report(self):
        if self.description:
            report = ModelReport(self.name, self.description)
        else:
            report = ModelReport(self.name, self._get_description())
        report.add("classification_report", self.crd)
        report.add("roc_auc", self.roc_auc)
        report.add("loss", self.scores[0])
        report.add("accuracy", self.scores[1])
        report.add("tpr", self.tpr)
        report.add("fpr", self.fpr)
        report.add("confusion_matrix", self.confusion_matrix)
        report.add("file", self.data_file)
        report.add("network_history_file", self.network_history)
        report.add("tokenizer_file", self.tokenizer_file)
        if self.X_train is not None:
            report.add("max_sequence_length", self.X_train.shape[1])
        report.add("embedding", self.embedding)
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

    def __init__(self, model_name, description=None):
        self.report = {}
        self.report["model_name"] = model_name
        self.report["description"] = description

    def add(self, key, value):
        """
        add value to report using key
        :param key:
        :param value:
        :return:
        """
        log.debug(f"key: {key} value: {value}")
        log.debug(f"key type: {type(key)} value type: {type(value)}")
        # if it's a list then we serialize to json string format so we can load it back later
        if isinstance(value, list) or isinstance(value, dict):
            self.report[key] = json.dumps(value)
        elif isinstance(value, np.ndarray):
            self.report[key] = json.dumps(value.tolist())
        else:
            self.report[key] = value

    def to_df(self):
        df = pd.DataFrame()
        return df.append(self.report, ignore_index=True)

    def save(self, report_file, append=False):
        # check to see if report file exisits, if so load it and append
        exists = os.path.isfile(report_file)
        if append and exists:
            print(f'Loading to append to: {report_file}')
            report_df = pd.read_csv(report_file, quotechar="'")
        else:
            report_df = pd.DataFrame()

        report_df = report_df.append(self.to_df(), ignore_index=True)
        print("Saving report file...")
        report_df.to_csv(report_file, index=False, quotechar="'")
        return report_df

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
        self.init = keras.initializers.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

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
