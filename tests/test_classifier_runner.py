from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from models.ClassifierRunner import ClassifierRunner, Keys, Status
from pandas import DataFrame
import logging
import pytest
from unittest.mock import MagicMock
from unittest import mock
import numpy as np

logging.basicConfig(format='%(asctime)s %(name)s.$(funcName)s %(levelname)s - %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)

class TestClassifierRunner(object):

    @pytest.fixture(scope="module")
    def get_train_x(self):
        train_x = DataFrame()
        train_x = train_x.append({"a":1, "b":1}, ignore_index=True)
        train_x = train_x.append({"a":50, "b":50}, ignore_index=True)
        return train_x


    @pytest.fixture(scope="module")
    def get_train_y(self):
        return DataFrame({"a": [1, 2]}).values.ravel()


    @pytest.fixture(scope="module")
    def get_knn(self):
        return KNeighborsClassifier(n_neighbors=1, n_jobs=1)

    # @pytest.fixture(scope="module")
    def mock_rn_fit(self, x_train, y_train):
        """
        mocking fit function to raise an exception
        :param x_train:
        :param y_train:
        :return:
        """
        raise Exception('mock - failed to predict model')


    # TODO: calling mock_rn_fit doesn't work here
    @pytest.fixture(scope="module")
    @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier.fit', side_effect=mock_rn_fit)
    @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
    def get_rn(self, rn, mock_fit_fail):
        log.info(f"setting mock rn fit function {mock_fit_fail}")
        rn.fit = mock_fit_fail
        return rn


    # @pytest.fixture(scope="module")
    # def get_model_failed():
    #     model = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    #     model.predict()



    def test_add_model(self, get_train_x, get_train_y, get_knn):
        train_x = get_train_x
        train_y = get_train_y
        log.info(f'shape of train_y: {train_y.shape}')

        cr = ClassifierRunner(write_to_csv=False)
        cr.addModel(get_knn, train_x, train_y, train_x, train_y)
        models = cr.models
        log.info(f'model length: {len(models)}')
        log.info(type(models.iloc[0][Keys.TRAIN_Y]))
        assert len(models) == 1, "length of models should be 1"
        assert models.iloc[0][Keys.MODEL], "model is null"
        assert models.iloc[0][Keys.MODEL_NAME], "model name is null"
        assert isinstance(models.iloc[0][Keys.TRAIN_X], DataFrame), "train_x name is null"
        assert isinstance(models.iloc[0][Keys.TRAIN_Y], np.ndarray), "train_y name is null"
        assert isinstance(models.iloc[0][Keys.TEST_X], DataFrame), "test_x name is null"
        assert isinstance(models.iloc[0][Keys.TEST_Y], np.ndarray), "test_y name is null"

        cr.addModel(get_knn, train_x, train_y, train_x, train_y)
        models = cr.models
        log.info(f'model length: {len(models)}')
        assert len(models) == 2, "length of models should be 2"
        assert models.iloc[1][Keys.MODEL], "model is null"
        assert models.iloc[1][Keys.MODEL_NAME], "model name is null"
        assert isinstance(models.iloc[1][Keys.TRAIN_X], DataFrame), "train_x name is null"
        assert isinstance(models.iloc[1][Keys.TRAIN_Y], np.ndarray), "train_y name is null"
        assert isinstance(models.iloc[1][Keys.TEST_X], DataFrame), "test_x name is null"
        assert isinstance(models.iloc[1][Keys.TEST_Y], np.ndarray), "test_y name is null"


    def test_run_one_model_success(self, get_train_x, get_train_y, get_knn):
        """
        :return:
        """
        cr = ClassifierRunner(write_to_csv=False)
        cr.addModel(get_knn, get_train_x, get_train_y, get_train_x, get_train_y, name="success case")
        report_df = cr.runModels()

        assert len(report_df) == 1, "report should have 1 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"


    # TODO: mocking fit function is not working but it is throwing an excpetion for the test to fail
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier.fit', side_effect=mock_rn_fit)
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
    def test_run_one_model_failed(self, get_rn, get_train_x, get_train_y):
        """
        :return:
        """
        # log.debug(f'mock_rn: {mock_rn}')
        # log.debug(f'mock_rn_fit: {mock_rn_fit}')
        rn = get_rn
        # rn.fit = mock_rn_fit
        log.debug(f'mocked rn? {rn}')
        log.debug(f'mocked rn.fit? {rn.fit}')
        cr = ClassifierRunner(write_to_csv=False)
        cr.addModel(rn, get_train_x, get_train_y, get_train_x, get_train_y, name="failed case")
        report_df = cr.runModels()

        log.info(f'error message: {report_df.iloc[0][Keys.MESSAGE]}')

        assert len(report_df) == 1, "report should have 1 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[0][Keys.MESSAGE]) > 0, "message should be set for FAILED models"


    # TODO: mocking fit function is not working but it is throwing an excpetion for the test to fail
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier.fit', side_effect=mock_rn_fit)
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
    def test_run_two_models(self, get_rn, get_knn, get_train_x, get_train_y):
        """
        first will pass
        second will fail
        :return:
        """
        cr = ClassifierRunner(write_to_csv=False)
        assert len(cr.report_df) == 0, "clean CR should have 0 length report"

        # first test - success
        knn = get_knn
        log.debug(f'test knn? {knn}')
        cr.addModel(knn, get_train_x, get_train_y, get_train_x, get_train_y, name="success case")

        # second test - fail
        rn = get_rn
        log.debug(f'mocked rn? {rn}')
        log.debug(f'mocked rn.fit? {rn.fit}')
        cr.addModel(rn, get_train_x, get_train_y, get_train_x, get_train_y, name="failed case")
        report_df = cr.runModels()

        assert len(report_df) == 2, "report should have 2 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"

        assert report_df.iloc[1][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[1][Keys.MESSAGE]) > 0, "message should be set for FAILED models"

    def test_dict_to_dict(self):
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

