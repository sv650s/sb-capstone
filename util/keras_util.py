from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import util.dict_util as du
import util.file_util as fu
import json
import logging
import os
from datetime import datetime
import pickle


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


def preprocess_file(data_df, feature_column, label_column, keep_percentile):
    """
    Preprocessing data file and create the right inputs for Keras models
        Generally using this for more advanced models like GRU, LSTM, CNN as we are embedding as part of this

        Features:
        * tokenize
        * pad features into sequence
        Labels:
        * one hot encoder

        split between training and testing

    :param data_df:
    :param feature_column:
    :param review_column:
    :param keep_percentile: percentile of feature length to keep - all features will be padded to this length
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
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(features_train)
    # tokenize both our training and test data
    train_sequences = t.texts_to_sequences(features_train)
    test_sequences = t.texts_to_sequences(features_test)

    print("Vocabulary size={}".format(len(t.word_counts)))
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

    def evaluate(self, X_test, y_test, verbose=1):
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
        y_predict_unencoded = unencode(self.y_predict)
        y_test_unencoded = unencode(self.y_test)


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


    def save(self, save_dir, append_report=False):
        description = self._get_description()
        print(f"description: {description}")

        self.model_file = f"{save_dir}/models/{description}-model.h5"
        self.network_history_file = f'{save_dir}/models/{description}-history.pkl'
        self.report_file = f"{save_dir}/reports/{datetime.now().strftime(DATE_FORMAT)}-dl_protype-report.csv"
        self.tokenizer_file = f'{save_dir}/models/dl-tokenizer.pkl'

        print(f"Saving model file: {self.model_file}")
        self.model.save(self.model_file)

        print(f"Saving network history file: {self.network_history_file}")
        pickle.dump(self.network_history, open(self.network_history_file, 'wb'))

        if self.tokenizer:
            log.info(f"Saving tokenizer file: {self.tokenizer_file}")
            pickle.dump(self.tokenizer, open(self.tokenizer_file, 'wb'))

        print(f"Saving to report file: {self.report_file}")
        report = self.get_report()
        report.save(self.report_file, append=append_report)


    def get_report(self):
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
        report.add("test_examples", self.X_test.shape[0])
        report.add("test_features", self.X_test.shape[1])
        report.add("train_examples", self.X_train.shape[0])
        report.add("train_features", self.X_train.shape[1])
        report.add("train_time_min", self.train_time_min)
        report.add("evaluate_time_min", self.train_time_min)
        report.add("predict_time_min", self.train_time_min)
        report.add("status", "success")
        report.add("status_date", datetime.now().strftime(TIME_FORMAT))

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


