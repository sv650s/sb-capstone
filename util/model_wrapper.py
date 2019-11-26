from util.constants import Keys, Status
from abc import ABC, ABCMeta, abstractmethod
from util.time_util import TimedReport, DATE_FORMAT, TIME_FORMAT
from datetime import datetime
import numpy as np
import traceback2
import sys
import json
import logging

log = logging.getLogger(__name__)


class AbstractModelWrapper(ABC):

    def __init__(self,
                 model: object,
                 library: str,
                 label_column: str,
                 description: str = None,
                 name: str = None,
                 file: str = None,
                 model_dir: str = "../models"
                 ):
        """
        Constructor

        Initializes:
            * TimedReport object
            * name
            * description
        :param model: model object - must have fit and transform functions
        :param label_column: name of column to get labels
        :param library: model library - ie, sklearn, pyspark
        :param description: description of model (optional)
        :param name: name of model. If not provided, will derive it from the classname (optional)
        :param file: data filename (optional)
        """
        self.report = TimedReport()
        self.model = model
        self.library = library
        self.name = name
        self.label_column = label_column
        self.file = file
        self.model_dir = model_dir

        if name is not None:
            self.name = name
        else:
            self.name = type(model).__name__
        log.info(f"derived name: {self.name}")

        if description is not None:
            self.description = f'{description}-{self.name}-{label_column}'
        else:
            self.description = f'{self.name}-{label_column}'

        self.status = Status.NEW
        self.status_date = None
        self.message = None
        self.y_predict = None
        self.classification_report = None
        self.confusion_matrix = None

        rdict = {
            Keys.MODEL_NAME: self.name,
            Keys.DESCRIPTION: self.description,
            Keys.LIBRARY: self.library,
            Keys.FILE: self.file,
            # Keys.TRAIN_EXAMPLES: train_row,
            # Keys.TRAIN_FEATURES: train_col,
            # Keys.TEST_EXAMPLES: test_row,
            # Keys.TEST_FEATURES: test_col,
            # Keys.PARAMETERS: json.dumps(parameters)
        }
        self.report.add_dict(rdict)

    def run(self, fit: bool = True, get_report: bool = True):
        """
        Fit and predict model
        :param train: call fit function on model. Set to false when using pre-trained/CV models. Default is True
        :return: report dictionary, y_predict
        """
        log.info('#' * 40)
        log.info(f'Running model: {str(self)}')
        log.info('#' * 40)

        try:
            if fit:
                self.report.start_timer(Keys.TRAIN_TIME_MIN)
                self.model = self._fit_model()
                self.report.end_timer(Keys.TRAIN_TIME_MIN)

            model_filename = self._get_model_filename()
            # model_filename = f'{self.model_dir}/{self.description}.jbl'
            self.report.record(Keys.MODEL_FILE, model_filename)

            self.report.start_timer(Keys.MODEL_SAVE_TIME_MIN)
            self._save_model(model_filename)
            # with open(model_filename, 'wb') as file:
            #     joblib.dump(self.model, model_filename)
            self.report.end_timer(Keys.MODEL_SAVE_TIME_MIN)

            self.report.start_timer(Keys.PREDICT_TIME_MIN)
            # self.y_predict = self.model.predict(self.x_test)
            self.y_predict = self._predict()
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

        if get_report:
            return self.y_predict, self.get_report_dict()

        return self.y_predict

    def __str__(self):
        """
        Override str method to summarize model
        :return:
        """
        return f'name: {self.name}\n' \
            f'\twith file: {self.file}\n' \
            f'\twith description: {self.description}\n' \
            f'\tstatus: {self.status}'
        # f'\twith parameters: {self.parameters}\n' \

    def get_report_dict(self, recalculate: bool = False) -> dict:
        """
        Creates a 1 level dictionary that summarizes this model so we can add it to DF later
        :return:
        """
        self._get_classification_report()
        self._get_confusion_matrix()
        # TODO: this is not working - calculate roc_auc for sklearn models
        # roc_auc, fpr, tpr = calculate_roc_auc(self.y_test, self.y_predict)
        # self.report.record(Keys.ROC_AUD, roc_auc)

        train_size, train_cols, test_size, test_cols = self._get_sizes()

        self.report.record(Keys.TRAIN_EXAMPLES, train_size)
        self.report.record(Keys.TRAIN_FEATURES, train_cols)
        self.report.record(Keys.TEST_EXAMPLES, test_size)
        self.report.record(Keys.TEST_FEATURES, test_cols)
        # Keys.PARAMETERS: json.dumps(parameters)

        self._add_to_report()

        return self.report.get_report_dict()

    def _get_classification_report(self, recalculate: bool = False):
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
        log.info("calculating classification report...")
        if self.classification_report is None or recalculate:
            self.report.start_timer(Keys.CR_TIME_MIN)
            self.classification_report = self._calculate_classification_report()
            self.report.end_timer(Keys.CR_TIME_MIN)
            if self.classification_report is not None:
                self.report.record(Keys.CR, json.dumps(self.classification_report))

    def _get_confusion_matrix(self, recalculate: bool = False):
        """
        Get confustion matrix and store in report in json format
        :return:
        """
        log.info("calculating confusion matrix...")
        if self.confusion_matrix is None or recalculate:
            self.report.start_timer(Keys.CM_TIME_MIN)
            self.confusion_matrix = self._calculate_confusion_matrix()
            self.report.end_timer(Keys.CM_TIME_MIN)
            if self.confusion_matrix is not None:
                self.report.record(Keys.CM, json.dumps(self.confusion_matrix.tolist()))

    def _get_model_filename(self) -> str:
        """
        gets the model file name
        :return: return filename model will be saved under
        """
        model_filename = f'{self.model_dir}/{self.description}.{self._get_model_file_extension()}'
        return model_filename

    def _add_to_report(self):
        """
        this is called at the end of get_report_dict before report
        is actually returned so child classes can add custom functionality
        here. Default implementation is it does nothing
        :return:
        """
        pass

    @abstractmethod
    def _fit_model(self):
        """
        run fit on model
        :return: fitted model
        """
        pass

    @abstractmethod
    def _predict(self):
        """
        run predict on the model
        :return: list of predictions
        """
        pass

    @abstractmethod
    def _get_model_file_extension(self):
        """
        extension of model file to save to - this should depends on what format
        model is saved under
        :return: model extension - ie, jbl, pkl, etc
        """
        pass

    @abstractmethod
    def _save_model(self, out_file):
        """
        save the model to out_file
        :param out_file: fully qualified file path to save the model
        :return:
        """
        pass

    @abstractmethod
    def _calculate_classification_report(self) -> dict:
        """
        creates sklearn like classification report dictionary
        :return: return classification report dictionary. Should return None if not available
        """
        pass

    @abstractmethod
    def _calculate_confusion_matrix(self) -> np.array:
        """
        creates sklearn like confusion matrix nparray
        :return: confusion matrix nparray. Should return None if not available
        """
        pass

    @abstractmethod
    def _get_sizes(self) -> int:
        """
        Gets sizes for training and test sets

        :return: training set size, number of test columns, test set size, # of test columns
        """
        pass

