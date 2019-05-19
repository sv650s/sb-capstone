#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.metrics import classification_report
from datetime import datetime
import logging
import pprint


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
    TOTAL_TIME_MIN = "total_time_min"
    STATUS = "status"
    STATUS_DATE = "status_date"
    MESSAGE = "message"
    MODEL_NAME = "model_name"
    MODEL = "model"
    MODEL_INDEX = "model_index"
    DATASET = "dataset"
    PARAMETERS = "param"
    TRAIN_X = "X_train"
    TRAIN_Y = "Y_train"
    TEST_X = "X_test"
    TEST_Y = "Y_test"




# set up logger
log = logging.getLogger(__name__)


class ClassifierRunner(object):
    """
    Class to help run various models
    """




    # expand classification report into dictionary
    # classifcation report is a 2 level dictionary. from documentation, it looks something like this
    # {'label 1': {'precision':0.5,
    #              'recall':1.0,
    #              'f1-score':0.67,
    #              'support':1},
    #  'label 2': { ... },
    #   ...
    # }
    @staticmethod
    def _add_dict_to_dict(target:dict, source:dict) -> dict:
        """
        target: dictionary to add to
        source: dictionary to add from
        ------
        return: dictionary with source added to target
        """
        for key, value in source.items():
            if isinstance(value, dict):
                # append key to dictionary keys
                for subkey, subvalue in value.items():
                    target[f'{key}_{subkey}'] = subvalue
            else:
                target[key] = value

        return target

    @staticmethod
    def _interpret_predictions(y_test, y_predict, report=None):
        """
        Run metrics on predictions

        y_test: true results
        y_predict: predictions from model
        results: dictionary to append results to
        ------
        return dictionary with report
        """

        if not report:
            report = {}

        c_report = classification_report(y_test, y_predict, output_dict = True)

        report = ClassifierRunner._add_dict_to_dict(report, c_report)

        return report

    @staticmethod
    def _model_fit_predict(model:object,
                           x_train:pd.DataFrame,
                           y_train:pd.DataFrame,
                           x_test:pd.DataFrame,
                           y_test:pd.DataFrame,
                           report:dict=None) -> (dict, pd.DataFrame):
            """
            Fit the model then run predict on it

            model: model to train with
            x_test: training input
            y_train: training classes
            x_test: test input
            y_test: result
            report: dict to append results to. Optional - will create this if there isn't one provided
            -----
            return tuple of predictions and dictionary with train time, predict_time total time
            """

            if not report:
                report = {}

            train_time_start = datetime.now()
            log.info(f'Start training: {train_time_start.strftime(TIME_FORMAT)}')
            model = model.fit(x_train, y_train)

            train_time_end = datetime.now()
            log.info(f'End training: {train_time_end.strftime(TIME_FORMAT)}')

    #         score = result.score(bag_x_test, bag_y_test)

            # calculate mean error score
            score_time_end = datetime.now()
            log.info(f'End Scoring: {score_time_end.strftime(TIME_FORMAT)}')

            # predictions
            predict_time_start = datetime.now()
            log.info(f'Start predict: {predict_time_start.strftime(TIME_FORMAT)}')
            y_predict = model.predict(x_test)
            predict_time_end = datetime.now()
            log.info(f'End predict: {predict_time_end.strftime(TIME_FORMAT)}')

            # calculate times
            train_time = train_time_end - train_time_start
            train_time_min = round(train_time.total_seconds() / 60, 1)
            log.info(f'Training time (min): {train_time_min}')

            score_time = score_time_end - train_time_end
            score_time_min = round(score_time.total_seconds() / 60, 1)
            log.info(f'Scoring time (min): {score_time_min}')

            predict_time = predict_time_end - score_time_end
            predict_time_min = round(predict_time.total_seconds() / 60,1 )
            log.info(f'Predict time (min): {predict_time_min}')

            train_examples, train_features = x_train.shape
            test_examples, test_features = x_test.shape

            report[Keys.TRAIN_EXAMPLES] = train_examples
            report[Keys.TRAIN_FEATURES] = train_features
            report[Keys.TEST_EXAMPLES] = test_examples
            report[Keys.TEST_FEATURES] = test_features
            report[Keys.TRAIN_TIME_MIN] = train_time_min
            report[Keys.SCORE_TIME_MIN] = score_time_min
            report[Keys.PREDICT_TIME_MIN] = predict_time_min
            report[Keys.TOTAL_TIME_MIN] = train_time_min + score_time_min + predict_time_min
            report = ClassifierRunner._interpret_predictions(y_test, y_predict, report)
            report[Keys.STATUS] = Status.SUCCESS
            report[Keys.STATUS_DATE] = datetime.now().strftime(TIME_FORMAT)

            return report, y_predict

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
    def __init__(self, write_to_csv = True, outfile = None):

        self.write_to_csv = write_to_csv
        if outfile:
            self.outfile = outfile
        else:
            self.outfile = f'{datetime.now().strftime("%Y-%m-%d")}-{__name__}-report.csv'
        self.models = pd.DataFrame()
        self.report_df = pd.DataFrame()
        log.info(f'Initializing {__name__}')
        log.info(f'write to csv: {self.write_to_csv}')
        log.info(f'outfile: {self.outfile}')


    def addModel(self, model:object, x_train:pd.DataFrame, y_train:pd.DataFrame,
                 x_test:pd.DataFrame, y_test:pd.DataFrame,
                 name:str = None, dataset:str = None, parameters:dict = None):
        """
        Add models to be executed
        :param model:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param name:
        :param dataset:
        :param parameters:
        :return:
        """

        model_dict = {
            Keys.MODEL: model,
            Keys.TRAIN_X: x_train,
            Keys.TRAIN_Y: y_train,
            Keys.TEST_X: x_test,
            Keys.TEST_Y: y_test,
            Keys.DATASET: dataset,
            Keys.PARAMETERS: parameters,
            Keys.STATUS: Status.NEW,
            Keys.STATUS_DATE: None
        }
        if name:
            model_dict[Keys.MODEL_NAME] = name
        else:
            model_dict[Keys.MODEL_NAME] = type(model).__name__

        log.debug(f'before adding models length: {len(self.models)}')
        log.debug(f'models : {pprint.pformat(self.models)}')
        log.debug(f'adding model [{pprint.pformat(model_dict)}]')
        self.models = self.models.append(model_dict, ignore_index=True)
        log.debug(f'models length: {len(self.models)}')


    def runAllModels(self) -> pd.DataFrame:
        """
        Runs all models configured for this runner
        :return:
        """
        for index, model in self.models.iterrows():
            report = self._runModel(index, model)
            self.models.iloc[index][Keys.STATUS] = report[Keys.STATUS]
            self.models.iloc[index][Keys.STATUS_DATE] = report[Keys.STATUS_DATE]
        return self.report_df


    def _runModel(self, index:int, model:pd.DataFrame) -> pd.DataFrame:
        log.info(f'Running model: {model[Keys.MODEL_NAME]}\n'
                 f'\twith data: {model[Keys.DATASET]}\n'
                 f'\tand parameters: {model[Keys.PARAMETERS]}')
        report = {
            Keys.MODEL_NAME: model[Keys.MODEL_NAME],
            Keys.DATASET: model[Keys.DATASET],
            Keys.PARAMETERS: model[Keys.PARAMETERS],
            Keys.MODEL_INDEX: index
        }
        try:
            report, _ = ClassifierRunner._model_fit_predict(model[Keys.MODEL],
                                                            model[Keys.TRAIN_X],
                                                            model[Keys.TRAIN_Y],
                                                            model[Keys.TEST_X],
                                                            model[Keys.TEST_Y],
                                                            report)
            report[Keys.STATUS] = Status.SUCCESS
        except Exception as e:
            log.error(str(e))
            report[Keys.STATUS] = Status.FAILED
            report[Keys.MESSAGE] = str(e)
        finally:
            report[Keys.STATUS_DATE] = datetime.now().strftime(TIME_FORMAT)
            self._record_results(report)
            log.info(f'Finished running model: {model[Keys.MODEL_NAME]}\n'
                     f'\twith data: {model[Keys.DATASET]}\n'
                     f'\tand parameters: {model[Keys.PARAMETERS]}'
                     f'\tstatus: {report[Keys.STATUS]}')

        return report


    def runNewModels(self, rerun_failed=False) -> pd.DataFrame:
        """
        Run models that we haven't executed yet
        :return:
        """
        for index, row in self.models.iterrows():
            log.debug(f'before filtering - model name {row[Keys.MODEL_NAME]} status {row[Keys.STATUS]}')


        if rerun_failed:
            filtered_models = self.models[(self.models[Keys.STATUS] == Status.FAILED) |
                                       (self.models[Keys.STATUS] == Status.NEW)]
        else:
            filtered_models = self.models[self.models[Keys.STATUS] == Status.NEW]

        log.debug(f'filtered model length {len(filtered_models)}')
        for index, row in filtered_models.iterrows():
            log.debug(f'after filtering - model name {row[Keys.MODEL_NAME]} status {row[Keys.STATUS]}')

        log.debug(f'filtered model count {len(filtered_models)}')
        log.debug(f'filtered models {filtered_models.head()}')

        for index, model in filtered_models.iterrows():
            report = self._runModel(index, model)
            self.models.iloc[index][Keys.STATUS] = report[Keys.STATUS]
            self.models.iloc[index][Keys.STATUS_DATE] = report[Keys.STATUS_DATE]
        return self.report_df





