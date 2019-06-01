#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.metrics import classification_report
from datetime import datetime
import logging
import traceback2
import sys
import pprint
from util.dict_util import add_dict_to_dict

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'


class Status(object):
    SUCCESS = "success"
    FAILED = "failed"
    NEW = "new"


class Keys(object):
    TRAIN_EXAMPLES = "train_examples"
    TRAIN_FEATURES = "train_features"
    TEST_EXAMPLES = "test_examples"
    TEST_FEATURES = "test_features"
    TRAIN_TIME_MIN = "train_time_min"
    SCORE_TIME_MIN = "score_time_min"
    PREDICT_TIME_MIN = "predict_time_min"
    SMOTE_TIME_MIN = "smote_time_min"
    FILE_LOAD_TIME_MIN = "file_load_time_min"
    TOTAL_TIME_MIN = "total_time_min"
    STATUS = "status"
    STATUS_DATE = "status_date"
    MESSAGE = "message"
    MODEL_NAME = "model_name"
    MODEL = "model"
    FILE = "file"
    DESCRIPTION = "description"
    PARAMETERS = "param"
    TRAIN_X = "X_train"
    TRAIN_Y = "Y_train"
    TEST_X = "X_test"
    TEST_Y = "Y_test"
    TIMER = "timer"


class Timer(object):

    def __init__(self):
        self.timer_dict = {}
        self.temp_dict = {}

    def start_timer(self, key: str):
        log.info(f'Start timer for: {key}')
        self.temp_dict[key] = datetime.now()

    def end_timer(self, key: str) -> float:
        """
        calculates the difference in minutes and return value
        :param key:
        :return:
        """
        log.info(f'End timer for: {key}')
        end_time = datetime.now()
        if key in self.temp_dict.keys():
            start_time = self.temp_dict[key]
            diff_mins = round((end_time - start_time).total_seconds() / 50, 2)
            self.timer_dict[key] = diff_mins
            log.info(f'Total time for {key}: {self.timer_dict[key]}')
            self.timer_dict[Keys.TOTAL_TIME_MIN] = self.get_total_time()
        else:
            log.info(f'No timer for: {key}')
            diff_mins = 0

        return diff_mins

    def get_time(self, key: str) -> float:
        """
        get time in minutes for particular event
        :param key:
        :return:
        """
        if key in self.timer_dict.keys:
            return self.timer_dict[key]
        return 0.

    def get_total_time(self) -> float:
        total = 0.0
        for key, val in self.timer_dict.items():
            total += val
        return total

    def merge_timers(self, other):
        if other:
            self.timer_dict.update(other.timer_dict)


class Model(object):
    """
    Encapsulates a model that we will be running
    """

    def __init__(self,
                 model: str,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 name: str = None,
                 description: str = None,
                 timer: Timer = None,
                 file: str = None,
                 parameters: dict = None,
                 ):
        """

        :param model:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param name:
        :param description:
        :param timer:
        :param file:
        :param parameters:
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.description = description
        self.file = file
        self.parameters = parameters
        self.status = Status.NEW
        self.timer = Timer()
        self.y_predict = None
        self.status_date = None
        self.message = None

        if timer:
            self.timer.merge_timers(timer)

        if name:
            self.name = name
        else:
            self.name = type(model).__name__

    def run(self):
        log.info(f'Running model: {str(self)}')

        try:

            self.timer.start_timer(Keys.TRAIN_TIME_MIN)
            model = self.model.fit(self.x_train, self.y_train)
            self.timer.end_timer(Keys.TRAIN_TIME_MIN)

            self.timer.start_timer(Keys.PREDICT_TIME_MIN)
            self.y_predict = model.predict(self.x_test)
            self.timer.end_timer(Keys.PREDICT_TIME_MIN)

            self.status = Status.SUCCESS

        except Exception as e:
            traceback2.print_exc(file=sys.stdout)
            log.error(str(e))
            self.status = Status.FAILED
            self.message = str(e)
        finally:
            self.status_date = datetime.now().strftime(TIME_FORMAT)
            log.info(f'Finished running model: {str(self)}')

        return self.get_report(), self.y_predict

    def __str__(self):
        """
        Override str method to summarize model
        :return:
        """
        return f'name: {self.name}\n' \
            f'\twith file: {self.file}\n' \
            f'\twith description: {self.description}\n' \
            f'\twith parameters: {self.parameters}\n' \
            f'\tstatus: {self.status}'

    def get_report(self) -> dict:
        """
        Creates a 1 level dictionary that summarizes this model so we can add it to DF later
        :return:
        """
        train_row, train_col = self.x_train.shape
        test_row, test_col = self.x_test.shape

        report = {
            Keys.MODEL_NAME: self.name,
            Keys.DESCRIPTION: self.description,
            Keys.FILE: self.file,
            Keys.PARAMETERS: self.parameters,
            Keys.TRAIN_EXAMPLES: train_row,
            Keys.TRAIN_FEATURES: train_col,
            Keys.TEST_EXAMPLES: test_row,
            Keys.TEST_FEATURES: test_col,
            Keys.STATUS: self.status,
            Keys.STATUS_DATE: self.status_date,
            Keys.MESSAGE: self.message,
        }
        report.update(self.timer.timer_dict)
        results = self.get_classification_report()
        report.update(results)
        return report

    def get_classification_report(self):
        """
        classification_report returns something like this:
            {'label 1': {'precision':0.5,
                 'recall':1.0,
                 'f1-score':0.67,
                 'support':1},
             'label 2': { ... },
              ...
            }

        This function will take the results and flatten it into one level so we can write it to a DF
        :return:
        """
        report = {}
        if len(self.y_predict) > 0:
            c_report = classification_report(self.y_test, self.y_predict, output_dict=True)
            report = add_dict_to_dict(report, c_report)
        return report


# set up logger
log = logging.getLogger(__name__)


class ClassifierRunner(object):
    """
    Class to help run various models
    """

    # @staticmethod
    # def _interpret_predictions(y_test, y_predict, report=None):
    #     """
    #     Run metrics on predictions
    #
    #     y_test: true results
    #     y_predict: predictions from model
    #     results: dictionary to append results to
    #     ------
    #     return dictionary with report
    #     """
    #
    #     if not report:
    #         report = {}
    #
    #     c_report = classification_report(y_test, y_predict, output_dict = True)
    #
    #     report = add_dict_to_dict(report, c_report)
    #
    #     return report

    # @staticmethod
    # def _model_fit_predict(model:object,
    #                        x_train:pd.DataFrame,
    #                        y_train:pd.DataFrame,
    #                        x_test:pd.DataFrame,
    #                        y_test:pd.DataFrame,
    #                        report:dict=None) -> (dict, pd.DataFrame):
    #         """
    #         Fit the model then run predict on it
    #
    #         model: model to train with
    #         x_test: training input
    #         y_train: training classes
    #         x_test: test input
    #         y_test: result
    #         report: dict to append results to. Optional - will create this if there isn't one provided
    #         -----
    #         return tuple of predictions and dictionary with train time, predict_time total time
    #         """
    #
    #         if not report:
    #             report = {}
    #
    #         train_time_start = datetime.now()
    #         log.info(f'Start training: {train_time_start.strftime(TIME_FORMAT)}')
    #         model = model.fit(x_train, y_train)
    #
    #         train_time_end = datetime.now()
    #         log.info(f'End training: {train_time_end.strftime(TIME_FORMAT)}')
    #
    # #         score = result.score(bag_x_test, bag_y_test)
    #
    #         # calculate mean error score
    #         score_time_end = datetime.now()
    #         log.info(f'End Scoring: {score_time_end.strftime(TIME_FORMAT)}')
    #
    #         # predictions
    #         predict_time_start = datetime.now()
    #         log.info(f'Start predict: {predict_time_start.strftime(TIME_FORMAT)}')
    #         y_predict = model.predict(x_test)
    #         predict_time_end = datetime.now()
    #         log.info(f'End predict: {predict_time_end.strftime(TIME_FORMAT)}')
    #
    #         # calculate times
    #         train_time = train_time_end - train_time_start
    #         train_time_min = round(train_time.total_seconds() / 60, 2)
    #         log.info(f'Training time (min): {train_time_min}')
    #
    #         score_time = score_time_end - train_time_end
    #         score_time_min = round(score_time.total_seconds() / 60, 2)
    #         log.info(f'Scoring time (min): {score_time_min}')
    #
    #         predict_time = predict_time_end - score_time_end
    #         predict_time_min = round(predict_time.total_seconds() / 60, 2)
    #         log.info(f'Predict time (min): {predict_time_min}')
    #
    #         train_examples, train_features = x_train.shape
    #         test_examples, test_features = x_test.shape
    #
    #         report[Keys.TRAIN_EXAMPLES] = train_examples
    #         report[Keys.TRAIN_FEATURES] = train_features
    #         report[Keys.TEST_EXAMPLES] = test_examples
    #         report[Keys.TEST_FEATURES] = test_features
    #         report[Keys.TRAIN_TIME_MIN] = train_time_min
    #         report[Keys.SCORE_TIME_MIN] = score_time_min
    #         report[Keys.PREDICT_TIME_MIN] = predict_time_min
    #         if Keys.FILE_LOAD_TIME_MIN in report.keys():
    #             report[Keys.TOTAL_TIME_MIN] = train_time_min + score_time_min + predict_time_min + \
    #                                             report[Keys.FILE_LOAD_TIME_MIN]
    #         else:
    #             report[Keys.TOTAL_TIME_MIN] = train_time_min + score_time_min + predict_time_min
    #         report = ClassifierRunner._interpret_predictions(y_test, y_predict, report)
    #         report[Keys.STATUS] = Status.SUCCESS
    #         report[Keys.STATUS_DATE] = datetime.now().strftime(TIME_FORMAT)
    #
    #         return report, y_predict

    # Use this method to record results for all test runs
    def _record_results(self, report: dict) -> pd.DataFrame:
        """
        appends report dictionary to results dataframe

        report: dictionary with results

        """
        self.report_df = self.report_df.append(report, ignore_index=True)
        if self.write_to_csv:
            self.report_df.to_csv(self.outfile, index=False)

    # constructor
    def __init__(self, cleanup=True, clean_failures=True, write_to_csv=True, outfile=None):

        self.write_to_csv = write_to_csv
        if outfile:
            self.outfile = outfile
        else:
            self.outfile = f'{datetime.now().strftime("%Y-%m-%d")}-{__name__}-report.csv'
        self.models = []
        self.report_df = pd.DataFrame()
        self.cleanup = cleanup
        self.clean_failures = clean_failures
        log.info(f'Initialized {__name__}\n\tcleanup={cleanup}\n\tclean_failures={clean_failures}'
                 f'\n\twrite_to_csv={write_to_csv}\n\toutfile={outfile}')

    def addModel(self, model: object):
        """
        Add models to be executed
        :param model:
        :return:
        """

        log.debug(f'before adding models length: {len(self.models)}')
        self.models.append(model)
        log.debug(f'new models length: {len(self.models)}')

    def run_models(self) -> pd.DataFrame:
        """
        Runs all models configured for this runner
        :return:
        """
        for model in self.models:
            model.run()
            report = model.get_report()
            self._record_results(report)
        return self.report_df

    # def runNewModels(self, rerun_failed=False) -> pd.DataFrame:
    #     """
    #     Run models that we haven't executed yet
    #     :return:
    #     """
    #     for index, row in self.models.iterrows():
    #         log.debug(f'before filtering - model name {row[Keys.MODEL_NAME]} status {row[Keys.STATUS]}')
    #
    #
    #     if len(self.models) > 0:
    #         if rerun_failed:
    #             filtered_models = self.models[(self.models[Keys.STATUS] == Status.FAILED) |
    #                                        (self.models[Keys.STATUS] == Status.NEW)]
    #         else:
    #             filtered_models = self.models[self.models[Keys.STATUS] == Status.NEW]
    #
    #         # log.debug(f'filtered model length {len(filtered_models)}')
    #         # for index, row in filtered_models.iterrows():
    #         #     log.debug(f'after filtering - model name {row[Keys.MODEL_NAME]} status {row[Keys.STATUS]}')
    #
    #         log.debug(f'filtered model count {len(filtered_models)}')
    #         # log.debug(f'filtered models {filtered_models.head()}')
    #
    #         for index, model in filtered_models.iterrows():
    #             log.info(f'Running {index+1} of {len(self.models)} models')
    #             report = self._runModel(model)
    #             self.models.at[index, Keys.STATUS] = report[Keys.STATUS]
    #             self.models.at[index, Keys.STATUS_DATE] = report[Keys.STATUS_DATE]
    #
    #     self._cleanModels()
    #
    #     return self.report_df
    #
    #
    # def _cleanModels(self):
    #     """
    #     remove references for each model so we can free up memory
    #     :return:
    #     """
    #     if self.cleanup and len(self.models) > 0:
    #         if self.clean_failures:
    #             m = self.models[(self.models[Keys.STATUS] == Status.SUCCESS) |
    #                             (self.models[Keys.STATUS] == Status.FAILED)]
    #         else:
    #             m = self.models[(self.models[Keys.STATUS] == Status.SUCCESS)]
    #
    #         self.models.drop(m.index, inplace=True)
