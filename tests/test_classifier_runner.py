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
    def train_x(self):
        train_x = DataFrame()
        train_x = train_x.append({"a":1, "b":1}, ignore_index=True)
        train_x = train_x.append({"a":50, "b":50}, ignore_index=True)
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


    # TODO: calling mock_rn_fit doesn't work here
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
        cr.addModel(success_model, train_x, train_y, train_x, train_y)
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

        cr.addModel(success_model, train_x, train_y, train_x, train_y)
        models = cr.models
        log.info(f'model length: {len(models)}')
        assert len(models) == 2, "length of models should be 2"
        assert models.iloc[1][Keys.MODEL], "model is null"
        assert models.iloc[1][Keys.MODEL_NAME], "model name is null"
        assert isinstance(models.iloc[1][Keys.TRAIN_X], DataFrame), "train_x name is null"
        assert isinstance(models.iloc[1][Keys.TRAIN_Y], np.ndarray), "train_y name is null"
        assert isinstance(models.iloc[1][Keys.TEST_X], DataFrame), "test_x name is null"
        assert isinstance(models.iloc[1][Keys.TEST_Y], np.ndarray), "test_y name is null"


    def test_run_one_model_success(self, train_x, train_y, success_model):
        """
        :return:
        """
        cr = ClassifierRunner(write_to_csv=False, cleanup=False)
        cr.addModel(success_model, train_x, train_y, train_x, train_y, name="success case")
        report_df = cr.runAllModels()
        models = cr.models

        assert len(report_df) == 1, "report should have 1 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"
        assert models.iloc[0][Keys.STATUS_DATE] is not None, "model status should be set"
        assert models.iloc[0][Keys.STATUS], "model status should be set"
        assert models.iloc[0][Keys.STATUS] == Status.SUCCESS, "model status should be SUCCESS"

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
        cr.addModel(rn, train_x, train_y, train_x, train_y, name="failed case")
        report_df = cr.runAllModels()
        models = cr.models
        log.info(f'model columns: {models.columns}')

        log.info(f'error message: {report_df.iloc[0][Keys.MESSAGE]}')

        assert len(report_df) == 1, "report should have 1 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[0][Keys.MESSAGE]) > 0, "message should be set for FAILED models"
        assert models.iloc[0][Keys.STATUS_DATE] is not None, "model status should be set"
        assert models.iloc[0][Keys.STATUS], "model status should be set"
        assert models.iloc[0][Keys.STATUS] == Status.FAILED, "model status FAILED"


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
        cr.addModel(knn, train_x, train_y, train_x, train_y, name="success case")

        # second test - fail
        rn = fail_model
        log.debug(f'mocked rn? {rn}')
        log.debug(f'mocked rn.fit? {rn.fit}')
        cr.addModel(rn, train_x, train_y, train_x, train_y, name="failed case")
        report_df = cr.runAllModels()

        assert len(report_df) == 2, "report should have 2 entry"
        assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"

        assert report_df.iloc[1][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[1][Keys.MESSAGE]) > 0, "message should be set for FAILED models"


    def test_run_new_models(self, fail_model, success_model, train_x, train_y):
        """
        add 1 model
        runAllModels
        add another model
        runNewModels

        should return 2 in the report
        :param get_rn:
        :param get_knn:
        :param get_train_x:
        :param get_train_y:
        :return:
        """
        cr = ClassifierRunner(write_to_csv=False)
        assert len(cr.report_df) == 0, "clean CR should have 0 length report"

        # first test - success
        knn = success_model
        log.debug(f'test knn? {knn}')
        cr.addModel(knn, train_x, train_y, train_x, train_y, name="success case")
        report_df = cr.runAllModels()
        assert report_df.iloc[0][Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"

        # second test - fail
        rn = fail_model
        log.debug(f'mocked rn? {rn}')
        log.debug(f'mocked rn.fit? {rn.fit}')
        cr.addModel(rn, train_x, train_y, train_x, train_y, name="failed case")
        report_df = cr.runNewModels()

        assert len(report_df) == 2, "report should have 2 entry"

        assert report_df.iloc[1][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[1][Keys.MESSAGE]) > 0, "message should be set for FAILED models"



    def test_rerun_failed_models(self, fail_model, success_model, train_x, train_y):
        """
        add 1 model
        runAllModels - should fail
        add another model
        runNewModels

        should return 2 in the report
        :param get_rn:
        :param get_knn:
        :param get_train_x:
        :param get_train_y:
        :return:
        """
        cr = ClassifierRunner(write_to_csv=False, cleanup=False)
        assert len(cr.report_df) == 0, "clean CR should have 0 length report"

        # first test - fail
        knn = fail_model
        log.debug(f'test knn? {knn}')
        cr.addModel(knn, train_x, train_y, train_x, train_y, name="first failed case")
        report_df = cr.runAllModels()
        assert report_df.iloc[0][Keys.STATUS] == Status.FAILED, "status should be FAILED"

        # second test - should run both models again
        rn = fail_model
        log.debug(f'mocked rn? {rn}')
        log.debug(f'mocked rn.fit? {rn.fit}')
        cr.addModel(rn, train_x, train_y, train_x, train_y, name="second failed case")
        report_df = cr.runNewModels(rerun_failed=True)

        for index, row in report_df.iterrows():
            log.debug(f'model name in report: {row[Keys.MODEL_NAME]}')

        assert len(report_df) == 3, "report should have 3 entry"

        assert report_df.iloc[2][Keys.STATUS] == Status.FAILED, "status should be FAILED"
        assert len(report_df.iloc[2][Keys.MESSAGE]) > 0, "message should be set for FAILED models"


    def test_default_cleanup(self, fail_model, success_model, train_y, train_x):
        cr = ClassifierRunner(write_to_csv=False)
        cr.addModel(success_model, train_x, train_y, train_x, train_y, name="first failed case")
        report_df = cr.runAllModels()
        models = cr.models

        assert len(report_df) == 1, "should get 1 report back"
        assert len(models) == 0, "should get 0 models back after cleanup"

        cr = ClassifierRunner(write_to_csv=False)
        cr.addModel(fail_model, train_x, train_y, train_x, train_y, name="first failed case")
        report_df = cr.runAllModels()
        models = cr.models

        assert len(report_df) == 1, "should get 1 report back"
        assert len(models) == 0, "should get 0 models back after cleanup"




    def test_cleanup_drop_failures(self, fail_model, success_model, train_y, train_x):
        cr = ClassifierRunner(write_to_csv=False, clean_failures=False)
        cr.addModel(fail_model, train_x, train_y, train_x, train_y, name="first failed case")
        report_df = cr.runAllModels()
        models = cr.models

        assert len(report_df) == 1, "should get 1 report back"
        assert len(models) == 1, "should get 0 models back after cleanup"


    def test_runs_with_no_models(self):
        cr = ClassifierRunner(write_to_csv=False)
        report_df = cr.runAllModels()
        assert len(report_df) == 0, "report length should be 0"
        report_df = cr.runNewModels()
        assert len(report_df) == 0, "report length should be 0"


    def test_multiple_models_with_clean(self, fail_model, success_model, train_x, train_y):
        cr = ClassifierRunner(write_to_csv=False)
        cr.addModel(success_model, train_x, train_y, train_x, train_y, name="second failed case")
        cr.addModel(fail_model, train_x, train_y, train_x, train_y, name="second failed case")
        cr.addModel(success_model, train_x, train_y, train_x, train_y, name="second failed case")
        report_df = cr.runAllModels()
        models = cr.models
        assert len(report_df) == 3, "should get 3 report back"
        assert len(models) == 0, "should get 0 models back after cleanup"


        cr.addModel(success_model, train_x, train_y, train_x, train_y, name="second failed case")
        cr.addModel(fail_model, train_x, train_y, train_x, train_y, name="second failed case")
        cr.addModel(success_model, train_x, train_y, train_x, train_y, name="second failed case")
        report_df = cr.runNewModels()
        models = cr.models
        assert len(report_df) == 6, "should get 3 report back"
        assert len(models) == 0, "should get 0 models back after cleanup"

