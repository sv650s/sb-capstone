from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from models.ClassifierRunner import ClassifierRunner, Keys, Status, Model
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
    def train_x(self):
        train_x = DataFrame()
        train_x = train_x.append({"a": 1, "b": 1}, ignore_index=True)
        train_x = train_x.append({"a": 50, "b": 50}, ignore_index=True)
        return train_x

    @pytest.fixture(scope="module")
    def train_y(self):
        return DataFrame({"a": [1, 2]}).values.ravel()

    @pytest.fixture(scope="module")
    def success_model(self):
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

    # TODO: calling mock_rn_fit doesn't work here for some reason, it's not raising the exception when being called
    @pytest.fixture(scope="module")
    @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier.fit', side_effect=mock_rn_fit)
    @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
    def fail_model(self, rn, mock_fit_fail):
        log.info(f"setting mock rn fit function {mock_fit_fail}")
        rn.fit = mock_fit_fail
        return rn

    def test_add_model(self, train_x, train_y, success_model):
        log.info(f'shape of train_y: {train_y.shape}')

        cr = ClassifierRunner(write_to_csv=False)
        model = Model(success_model, train_x, train_y, train_x, train_y)
        cr.addModel(model)
        models = cr.models
        log.info(f'model length: {len(models)}')
        log.info(type(models[0].y_train))
        assert len(models) == 1, "length of models should be 1"
        assert models[0].model, "model is null"
        assert models[0].name, "model name is null"
        assert isinstance(models[0].x_train, DataFrame), "train_x name is null"
        assert isinstance(models[0].y_train, np.ndarray), "train_y name is null"
        assert isinstance(models[0].x_test, DataFrame), "test_x name is null"
        assert isinstance(models[0].y_test, np.ndarray), "test_y name is null"

        model = Model(success_model, train_x, train_y, train_x, train_y)
        cr.addModel(model)
        models = cr.models
        log.info(f'model length: {len(models)}')
        assert len(models) == 2, "length of models should be 2"
        assert models[1].model, "model is null"
        assert models[1].name, "model name is null"
        assert isinstance(models[1].x_train, DataFrame), "train_x name is null"
        assert isinstance(models[1].y_train, np.ndarray), "train_y name is null"
        assert isinstance(models[1].x_test, DataFrame), "test_x name is null"
        assert isinstance(models[1].y_test, np.ndarray), "test_y name is null"

    def test_run_one_model_success(self, train_x, train_y, success_model):
        """
        :return:
        """
        cr = ClassifierRunner(write_to_csv=False, cleanup=False)
        model = Model(success_model, train_x, train_y, train_x, train_y, name="success case")
        cr.addModel(model)
        report_df = cr.run_models()
        models = cr.models

        assert len(report_df) == 1, "report should have 1 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"
        assert models[0].status_date is not None, "model status should be set"
        assert models[0].status, "model status should be set"
        assert models[0].status == Status.SUCCESS, "model status should be SUCCESS"

    # TODO: mocking fit function is not working but it is throwing an excpetion for the test to fail
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier.fit', side_effect=mock_rn_fit)
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
    def test_run_one_model_failed(self, fail_model, train_x, train_y):
        """
        :return:
        """
        # log.debug(f'mock_rn: {mock_rn}')
        # log.debug(f'mock_rn_fit: {mock_rn_fit}')
        rn = fail_model
        # rn.fit = mock_rn_fit
        log.debug(f'mocked rn? {rn}')
        log.debug(f'mocked rn.fit? {rn.fit}')
        cr = ClassifierRunner(write_to_csv=False, cleanup=False)
        model = Model(rn, train_x, train_y, train_x, train_y, name="failed case")
        cr.addModel(model)
        report_df = cr.run_models()
        models = cr.models

        log.info(f'error message: {report_df.iloc[0][Keys.MESSAGE]}')

        assert len(report_df) == 1, "report should have 1 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[0][Keys.MESSAGE]) > 0, "message should be set for FAILED models"
        assert models[0].status_date is not None, "model status should be set"
        assert models[0].status, "model status should be set"
        assert models[0].status == Status.FAILED, "model status FAILED"

    # TODO: mocking fit function is not working but it is throwing an excpetion for the test to fail
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier.fit', side_effect=mock_rn_fit)
    # @mock.patch('sklearn.neighbors.RadiusNeighborsClassifier')
    def test_run_two_models(self, fail_model, success_model, train_x, train_y):
        """
        first will pass
        second will fail
        :return:
        """
        cr = ClassifierRunner(write_to_csv=False, cleanup=False)
        assert len(cr.report_df) == 0, "clean CR should have 0 length report"

        # first test - success
        knn = success_model
        log.debug(f'test knn? {knn}')
        model = Model(knn, train_x, train_y, train_x, train_y, name="success case")
        cr.addModel(model)

        # second test - fail
        rn = fail_model
        log.debug(f'mocked rn? {rn}')
        log.debug(f'mocked rn.fit? {rn.fit}')
        model = Model(rn, train_x, train_y, train_x, train_y, name="failed case")
        cr.addModel(model)
        report_df = cr.run_models()

        assert len(report_df) == 2, "report should have 2 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"

        assert report_df.iloc[1][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[1][Keys.MESSAGE]) > 0, "message should be set for FAILED models"




