#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import logging
import traceback2
import sys
from util.dict_util import add_dict_to_dict
from util.time_util import Keys, TimedReport, Status, TIME_FORMAT, DATE_FORMAT
from sklearn.externals import joblib
import numpy as np

# set up logger
log = logging.getLogger(__name__)
MODEL_DIR = "../models"

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
                 class_column: str,
                 name: str = None,
                 description: str = None,
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
        :param file:
        :param parameters:
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.class_column = class_column
        self.file = file
        self.parameters = parameters
        self.status = Status.NEW
        self.y_predict = None
        self.status_date = None
        self.message = None

        self.report = TimedReport()
        # if report and isinstance(report, TimedReport):
        #     # outside program may have certain things already timed and recorded, we are going to merge them here
        #     self.report.merge_reports(report)
        # elif report:
        #     raise Exception("report is not a TimedReport")

        if name:
            self.name = name
        else:
            self.name = type(model).__name__
        self.description = f'{description}-{name}-{class_column}'

        train_row, train_col = self.x_train.shape
        test_row, test_col = self.x_test.shape

        rdict = {
            Keys.MODEL_NAME: self.name,
            Keys.DESCRIPTION: self.description,
            Keys.FILE: self.file,
            Keys.TRAIN_EXAMPLES: train_row,
            Keys.TRAIN_FEATURES: train_col,
            Keys.TEST_EXAMPLES: test_row,
            Keys.TEST_FEATURES: test_col,
            Keys.PARAMETERS: parameters
        }
        self.report.add_dict(rdict)
        # do this here so ti doesn't get flattened

    def run(self):
        log.info(f'Running model: {str(self)}')

        try:

            self.report.start_timer(Keys.TRAIN_TIME_MIN)
            self.model = self.model.fit(self.x_train, self.y_train)
            self.report.end_timer(Keys.TRAIN_TIME_MIN)

            # TODO: add logic for CV's
            model_filename = f'{MODEL_DIR}/{self.description}.jbl'
            self.report.record(Keys.MODEL_FILE, model_filename)
            self.report.start_timer(Keys.MODEL_SAVE_TIME_MIN)
            with open(model_filename, 'wb') as file:
                joblib.dump(self.model, model_filename)
            self.report.end_timer(Keys.MODEL_SAVE_TIME_MIN)

            self.report.start_timer(Keys.PREDICT_TIME_MIN)
            self.y_predict = self.model.predict(self.x_test)
            self.report.end_timer(Keys.PREDICT_TIME_MIN)

            self.report.record(Keys.STATUS, Status.SUCCESS)

        except Exception as e:
            traceback2.print_exc(file=sys.stdout)
            log.error(str(e))
            self.report.record(Keys.STATUS, Status.FAILED)
            self.report.record(Keys.MESSAGE, str(e))
        finally:
            self.report.record(Keys.STATUS_DATE, datetime.now().strftime(TIME_FORMAT))
            log.info(f'Finished running model: {str(self)}')

        return self.get_report_dict(), self.y_predict

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

    def get_report_dict(self) -> dict:
        """
        Creates a 1 level dictionary that summarizes this model so we can add it to DF later
        :return:
        """
        self.get_classification_report()
        self.get_confusion_matrix()
        return self.report.get_report_dict()

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
        log.debug(f'y_predict {self.y_predict}')
        log.debug(f'y_test {self.y_test}')
        if len(self.y_predict) > 0 and len(self.y_test) > 0:
            log.debug(f'getting classificaiton report for {self}')
            c_report = classification_report(self.y_test, self.y_predict, output_dict=True)
            self.report.add_and_flatten_dict(c_report)

    def get_confusion_matrix(self):
        """
        Get confustion matrix and store in report
        :return:
        """
        if len(self.y_predict) > 0 and len(self.y_test) > 0:
            log.debug(f'getting confusion matrix for {self}')
            cm = confusion_matrix(self.y_test, self.y_predict)
            self.report.record(Keys.CM, cm)


