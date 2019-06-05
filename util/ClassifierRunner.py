#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.metrics import classification_report
from datetime import datetime
import logging
import traceback2
import sys
from util.dict_util import add_dict_to_dict

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
# set up logger
log = logging.getLogger(__name__)


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
    VECTORIZER_TIME_MIN = "vectorizer_time_min"
    LDA_TIME_MIN = "lda_time_min"
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
            # TOOD: refactor here later. I don't like this dependency. Want to make timer more generic
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


class ClassifierRunner(object):
    """
    Class to help run various models
    """

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
