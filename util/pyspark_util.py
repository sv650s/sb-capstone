import numpy as np
import pyspark as pyspark
import logging
import sklearn.metrics as skmetrics
import sklearn.utils as skutils

from util.constants import Keys
from util.model_wrapper import AbstractModelWrapper

log = logging.getLogger(__name__)


def show_df(df: pyspark.sql.DataFrame,
            columns: list,
            rows: int = 10,
            sample=False,
            truncate=True):
    """
    Prints out number of rows in pyspark df

    :param df:  pyspark dataframe
    :param columns: list of columns to print
    :param rows: how many rows to print - default 10
    :param sample: should we sample - default False
    :param truncate: truncate output - default True
    :return:
    """
    if sample:
        sample_percent = min(rows / df.count(), 1.0)
        log.info(f'sampling percentage: {sample_percent}')
        df.select(columns).sample(False, sample_percent, seed=1).show(rows, truncate=truncate)
    else:
        df.select(columns).show(rows, truncate=truncate)


def classification_report(test_df: pyspark.sql.DataFrame,
                          truth_column: str,
                          prediction_column: str,
                          n_classes: int) -> dict:
    """
    Calculates the same classification report that skelarn would without converting labels
    and predictions to pandas or numpy

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    F1 = 2 * (precision * recall) / (precision + recall)

    weight average = sum(metric) / total support


    :param test_df: pyspark dataframe
    :param truth_column: name of truth column
    :param prediction_column: name of prediction column
    :param n_classes: number of classes
    :return:
    """
    log.info(f'truth_column: {truth_column} prediction_column: {prediction_column} classes {n_classes}')
    precisions = []
    recalls = []
    f1s = []
    weighted_precisions = []
    weighted_recalls = []
    weighted_f1s = []

    total_support = test_df.count()

    dr_dict = {}

    print()
    for i in np.arange(1, n_classes + 1):
        support = test_df.filter(f'{truth_column} == {i}').count()

        predicted = test_df.filter(f'{prediction_column} == {i}')
        tp = predicted.filter(f'{truth_column} == {prediction_column}').count()
        fp = predicted.count() - tp

        predicted_count = predicted.count()
        precision = 0 if predicted_count == 0 else tp / predicted.count()
        precisions.append(precision)
        weighted_precisions.append(precision * support)

        recall = tp / support
        recalls.append(recall)
        weighted_recalls.append(recall * support)

        f1 = 0 if precision == 0 and recall == 0 else 2 * (precision * recall) / (precision + recall)
        f1s.append(f1)
        weighted_f1s.append(f1 * support)
        print(f'Class {i} precision: {precision} recall: {recall} f1: {f1} support: {support}')

        d = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
        dr_dict[str(i)] = d

    print(f'Total Support: {total_support}')

    accuracy = test_df.filter("star_rating == prediction").count() / total_support
    dr_dict["accuracy"] = accuracy
    print(f'Accuracy: {accuracy}')

    macro_avg_precision = sum(precisions) / len(precisions)
    macro_avg_recall = sum(recalls) / len(recalls)
    macro_avg_f1 = sum(f1s) / len(f1s)

    dr_dict["macro avg"] = {
        "precision": macro_avg_precision,
        "recall": macro_avg_recall,
        "f1-score": macro_avg_f1,
        "support": total_support
    }

    print(f'Macro Avg Precision: {macro_avg_precision} Recall: {macro_avg_recall} F1: {macro_avg_f1}')

    weighted_avg_precision = sum(weighted_precisions) / total_support
    weighted_avg_recall = sum(weighted_recalls) / total_support
    weighted_avg_f1 = sum(weighted_f1s) / total_support

    dr_dict["weighted avg"] = {
        "precision": weighted_avg_precision,
        "recall": weighted_avg_recall,
        "f1-score": weighted_avg_f1,
        "support": total_support
    }

    print(f'Weighted Avg Precision: {weighted_avg_precision} Recall: {weighted_avg_recall} F1: {weighted_avg_f1}')

    return dr_dict


def confusion_matrix(test_df: pyspark.sql.DataFrame,
                     truth_column: str,
                     prediction_column: str,
                     n_classes: int):
    """
    Calculates confusiion matrix like sklearn would using pyspark dataframe

    :param test_df: pyspark dataframe with labels and predictions
    :param truth_column: column name with the truth
    :param prediction_column: column name for prediction
    :param n_classes: number of classes in predictions
    :return: np array with confusion matrix
    """
    cm = []
    for i in np.arange(1, n_classes + 1):
        current = []
        for j in np.arange(1, n_classes + 1):
            count = test_df.filter(f'({truth_column} == {i}) AND ({prediction_column} == {j})').count()

            current.append(count)
        cm.append(current)
    return np.asarray(cm)


def compute_class_weights(train_df: pyspark.sql.DataFrame,
                          label_column: str,
                          n_classes: int) -> np.array:
    """
    computes class weights based on pyspark dataframe

    :param train_df: pyspark dataframe with training data
    :param label_column: column name with labels
    :param n_classes: number of classes to classify
    :return: nparray with a weight for each class of length n_classes
    """
    # custom calculate class weights
    n_samples = train_df.count()
    class_weights = []

    for i in np.arange(1, n_classes + 1):
        class_samples = train_df.filter(f"{label_column} == {i}").count()
        class_weights.append(n_samples / (n_classes * class_samples))

    log.info(f'calculated class weights: {class_weights}')
    return class_weights


class PysparkModel(AbstractModelWrapper):
    """
    Wrapper for Pyspark model
    """

    PREDICTION_COL = "prediction"

    def __init__(self,
                 model,
                 train_df: pyspark.sql.DataFrame,
                 test_df: pyspark.sql.DataFrame,
                 label_column: str,
                 feature_column: str,
                 n_classes: int,
                 pipeline: object,
                 name: str = None,
                 file: str = None,
                 description: str = "pyspark",
                 model_dir: str = None,
                 ):
        """
        Constructor

        :param model: pyspark model
        :param train_df:
        :param test_df:
        :param label_column:
        :param feature_column:
        :param description: description of this model
        :param file: data filename
        """
        super(PysparkModel, self).__init__(model,
                                           "pyspark",
                                           label_column,
                                           description,
                                           name,
                                           file,
                                           model_dir)

        self.train_df = train_df
        self.test_df = test_df
        self.feature_column = feature_column
        self.n_classes = n_classes
        self.pipeline = pipeline

    def _fit_model(self) -> object:
        """
        fit and transform our model with training data
        :return:
        """
        self.model = self.model.fit(self.train_df)
        self.predict_train = self.model.transform(self.train_df)
        return self.model

    def _predict(self) -> pyspark.sql.DataFrame:
        """
        Runs transform on test data
        :return:
        """
        self.predict_test = self.model.transform(self.test_df)
        log.debug(self.predict_test.printSchema())
        return self.predict_test

    def _get_model_file_extension(self):
        """
        extension when saving model files
        :return:
        """
        return "pyspark"

    def _save_model(self, out_file: str):
        """
        saves the pyspark model
        :param out_file:
        :return:
        """
        log.info(f'Saving model to file: {out_file}')
        self.model.write().overwrite().save(out_file)

        pipeline_file = out_file.replace(f'.{self._get_model_file_extension()}',
                                         '.pipeline')
        log.info(f'Saving pipeline to file: {pipeline_file}')
        self.pipeline.write().overwrite().save(pipeline_file)
        self.report.record(Keys.PIPELINE_FILE, pipeline_file)

    def _calculate_classification_report(self):
        """
        creates a classification report dictionary based on pyspark dataframe
        from test data
        :return: dict with CR
        """
        # return classification_report(self.predict_test,
        #                              self.label_column,
        #                              PysparkModel.PREDICTION_COL,
        #                              self.n_classes)
        return skmetrics.classification_report(
            self.predict_test.select(self.label_column).toPandas()[self.label_column].tolist(),
            self.predict_test.select(PysparkModel.PREDICTION_COL).toPandas()[PysparkModel.PREDICTION_COL].tolist(),
            output_dict=True
        )

    def _calculate_confusion_matrix(self) -> np.array:
        """
        creates sklearn like confusion matrix
        :return: nparray representation of confusion matrix
        """
        # return confusion_matrix(self.predict_test,
        #                         self.label_column,
        #                         PysparkModel.PREDICTION_COL,
        #                         self.n_classes)

        return skmetrics.confusion_matrix(
            self.predict_test.select(self.label_column).toPandas()[self.label_column].tolist(),
            self.predict_test.select(PysparkModel.PREDICTION_COL).toPandas()[PysparkModel.PREDICTION_COL].tolist()
        )

    def _get_sizes(self) -> int:
        return self.train_df.count(), \
                self.train_df.select(self.feature_column).limit(1).toPandas().features.values[0].size, \
                self.test_df.count(), \
                self.test_df.select(self.feature_column).limit(1).toPandas().features.values[0].size

