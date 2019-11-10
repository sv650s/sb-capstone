from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import pandas as pd
import numpy as np
import logging
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D, MaxPool1D, Embedding
from tensorflow.keras.utils import model_to_dot

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from util.keras_util import ModelReport, ModelWrapper
import util.keras_util as ku


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestModelReport(object):


    def test_add_objects(self):
        """
        Test to make sure dics, lists, and ndarrays are jsonized before saving
        :return:
        """
        d = {"a": 1}
        l = [1, 2, 3]
        npa = np.array([1, 2, 3,])

        report = ModelReport("model_name")

        report.add("dict", d)
        assert isinstance(report.get("dict"), str), "returned object is not a string"
        assert json.loads(report.get("dict")), "dict not saved in valid json format"
        assert json.loads(report.get("dict"))["a"] == 1, "dict value should be 1"

        report.add("list", l)
        assert isinstance(report.get("list"), str), "returned object is not a string"
        assert json.loads(report.get("list")), "list not saved in valid json format"
        assert len(json.loads(report.get("list"))) == 3, "list length should be 3"

        report.add("np.array", npa)
        assert isinstance(report.get("np.array"), str), "returned object is not a string"
        assert json.loads(report.get("np.array")), "np.array not saved in valid json format"
        assert len(json.loads(report.get("np.array"))) == 3, "returned np.array does not have same length"

def test_load_model_report(shared_datadir):
    """
    Test loading a ModelReport from a file and make sure the final report has all columns in it
    :param shared_datadir:
    :return:
    """
    filename = f'{shared_datadir}/2019-11-dl_prototype-report.csv'
    report = ModelReport.load_report(filename, 0)
    assert report is not None, f'Report is None {report}'

    s = pd.read_csv(filename, quotechar=",").iloc[0]
    assert isinstance(s, pd.Series), f"s is not a Series: {type(s)}"
    assert len(s) >= 1, f"Dataframe length is less than 1 {len(s)}"
    log.debug(f'report keys: {report.report.keys()}')
    for col, value in s.items():
        assert col in list(report.report.keys()), f'{col} missing in ModelReport'




# Testing ModelWrapper - datadir does not allow this to be a class for some reason
@pytest.fixture()
def feature_data():
    df = pd.DataFrame({"a":
                           [ 0, 1, 2, 5, 10],
                       "b":
                           [3, 4, 5, 4, 6]
                       })
    return df

@pytest.fixture()
def label_data():
    return np.array([[0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0]])

def test_model_wrapper(datadir, feature_data, label_data):
    """
    Test to make sure that we can save and load model_wrapper
    :return:
    """


    model = Sequential()
    model.add(Dense(1, input_shape = (feature_data.shape[1],), kernel_initializer="glorot_uniform"))
    model.add(Activation('softmax'))
    model.add(Dense(5, activation="relu"))

    model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data)

    mw = ModelWrapper(model,
                      "model_name",
                      "architecture",
                      "feature_set_name",
                      "label_columns",
                      "data/data_file_name.csv",
                      embed_size=0,
                      )


    network_history = mw.fit(X_train, y_train,
                             batch_size=1,
                             epochs=1,
                             verbose=1
                             )

    mw.evaluate(X_test, y_test)
    mw.save(datadir)

    report = pd.read_csv(ModelWrapper.get_report_file_name(datadir))

    cols = ["classification_report",
            "roc_auc",
            "loss",
            "accuracy",
            "confusion_matrix",
            "file",
            "tokenizer_file",
            "max_sequence_length",
            "embedding",
            "model_file",
            "model_json_file",
            "weights_file",
            "sampling_type",
            "epochs",
            "test_examples",
            "test_features",
            "train_examples",
            "train_features",
            "train_time_min",
            "evaluate_time_min",
            "predict_time_min",
            "status",
            "status_date"]

    for col in cols:
        assert col in report.columns, f"report missing column: {col}"
        # check to make sure we have values everywhere
        assert len(report[col]) > 0, f'{col} value is 0'


@pytest.fixture()
def data_df():
    return pd.DataFrame({
        "review_body": [
            "this product is great", "this product is ok", "this product pretty bad", "the worst", "almost perfect"
        ],
        "star_rating": [
            5, 3, 2, 1, 4
        ]})



def test_preprocessing_sampling(datadir, data_df):

    sampler = RandomUnderSampler()

    X_train, X_test, y_train, y_test, t = ku.preprocess_file(data_df, "review_body", "star_rating", report_dir=f'{datadir}/reports', sampler=sampler)
    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(Dense(1, input_shape = (X_train.shape[1],), kernel_initializer="glorot_uniform"))
    model.add(Activation('softmax'))
    model.add(Dense(5, activation="relu"))

    model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])

    network = model.fit(X_train, y_train)

    assert network is not None, "network is none"


def test_preprocessing_nosampling(datadir, data_df):

    X_train, X_test, y_train, y_test, t = ku.preprocess_file(data_df, "review_body", "star_rating", report_dir=f'{datadir}/reports')
    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(Dense(1, input_shape = (X_train.shape[1],), kernel_initializer="glorot_uniform"))
    model.add(Activation('softmax'))
    model.add(Dense(5, activation="relu"))

    model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])

    network = model.fit(X_train, y_train)

    assert network is not None, "network is none"




