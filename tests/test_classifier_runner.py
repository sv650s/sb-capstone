from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from models.ClassifierRunner import ClassifierRunner, Keys, Status
from pandas import DataFrame
import logging
import pytest
from unittest.mock import MagicMock
from unittest import mock

logging.basicConfig(format='%(asctime)s %(name)s.$(funcName)s %(levelname)s - %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)




@pytest.fixture(scope="module")
def get_train_x():
    train_x = DataFrame()
    train_x = train_x.append({"a":1, "b":1}, ignore_index=True)
    train_x = train_x.append({"a":50, "b":50}, ignore_index=True)
    return train_x


@pytest.fixture(scope="module")
def get_train_y():
    return DataFrame({"a": [1, 2]}).values.ravel()


@pytest.fixture(scope="module")
def get_knn():
    return KNeighborsClassifier(n_neighbors=1, n_jobs=1)


@pytest.fixture(scope="module")
@mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
def get_rn():
    return RadiusNeighborsClassifier(radius=1.0, n_jobs=1)


# @pytest.fixture(scope="module")
# def get_model_failed():
#     model = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
#     model.predict()


def mock_rn_fit(x_train, y_train):
    """
    mocking fit function to raise an exception
    :param x_train:
    :param y_train:
    :return:
    """
    raise Exception('vince - failed to predict model')



def test_add_model(get_train_x, get_train_y, get_knn):
    train_x = get_train_x
    train_y = get_train_y
    log.info(f'shape of train_y: {train_y.shape}')

    cr = ClassifierRunner(write_to_csv=False)
    cr.addModel(get_knn, train_x, train_y, train_x, train_y)
    models = cr.models
    log.info(f'model length: {len(models)}')
    assert len(models) == 1, "length of models should be 1"
    assert models[0][Keys.MODEL], "model is null"
    assert models[0][Keys.MODEL_NAME], "model name is null"
    assert isinstance(models[0][Keys.TRAIN_X], DataFrame), "train_x name is null"
    assert isinstance(models[0][Keys.TRAIN_Y], DataFrame), "train_y name is null"
    assert isinstance(models[0][Keys.TEST_X], DataFrame), "test_x name is null"
    assert isinstance(models[0][Keys.TEST_Y], DataFrame), "test_y name is null"


    cr.addModel(get_knn, train_x, train_y, train_x, train_y)
    models = cr.models
    log.info(f'model length: {len(models)}')
    assert len(models) == 2, "length of models should be 2"
    assert models[1][Keys.MODEL], "model is null"
    assert models[1][Keys.MODEL_NAME], "model name is null"
    assert isinstance(models[1][Keys.TRAIN_X], DataFrame), "train_x name is null"
    assert isinstance(models[1][Keys.TRAIN_Y], DataFrame), "train_y name is null"
    assert isinstance(models[1][Keys.TEST_X], DataFrame), "test_x name is null"
    assert isinstance(models[1][Keys.TEST_Y], DataFrame), "test_y name is null"


def test_run_one_model(get_train_x, get_train_y, get_knn):
    """
    :return:
    """
    cr = ClassifierRunner(write_to_csv=False)
    cr.addModel(get_knn, get_train_x, get_train_y, get_train_x, get_train_y, name="success case")
    report_df = cr.runModels()

    assert len(report_df) == 1, "report should have 1 entry"
    assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"


@mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
def test_run_two_model(mock_rn):
    """
    :return:
    """
    rn = mock_rn
    rn.fit=mock_rn_fit
    #rn = RadiusNeighborsClassifier(radius=1.0, n_jobs=1)
    log.info(f'mocked rn? {rn}')
    log.info(f'mocked rn.fit? {rn.fit}')
    cr = ClassifierRunner(write_to_csv=False)
    cr.addModel(rn, get_train_x, get_train_y, get_train_x, get_train_y, name="failed case")
    report_df = cr.runModels()

    log.info(f'error message: {report_df.iloc[0][Keys.MESSAGE]}')

    assert len(report_df) == 1, "report should have 1 entry"
    assert report_df.iloc[0][Keys.STATUS] == Status.FAILED, "status should be FAILED"
    assert len(report_df.iloc[0][Keys.MESSAGE]) > 0, "message should be set for FAILED models"


def test_dict_to_dict():
    """
    Test to see if this flattens a dictionary
    :return:
    """
    target = {"a": 0}
    source = {"b": 1, "c": {"1": 2, "2": 3}}
    outcome = ClassifierRunner._add_dict_to_dict(target, source)
    key_num = 4
    assert len(outcome.keys()) == key_num, \
        f"number of keys should be {key_num}"

