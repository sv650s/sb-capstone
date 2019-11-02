import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)



def calculate_metric(data:pd.DataFrame, column_name = "eval_metric") -> pd.DataFrame:
    """
    Calculates the geometric mean in the following manner so we can use one metric to evalute our models
        recall - star rating 2, 3, 4
        precision - star rating 1, and 5

    :param data: report df - must have 1_precision, 2_recall, 3_recall, 4_recall, and 5_precision columns
    :param column_name: name of column to put metric in. default eval_metric
    :return:
    """
    return _harmonic_mean(data, column_name)

def _harmonic_mean(data:pd.DataFrame, column_name) -> pd.DataFrame:
    """
    Calcuates one metric score using harmonic mean using precision for star 1 and 5, recall for stars 2, 3, 4

    :param data:
    :param column_name:
    :return:
    """
    data[column_name] = 5 / (1/data["1_precision"] +
                                 1/data["2_recall"] +
                                 1/data["3_recall"] +
                                 1/data["4_recall"] +
                                 1/data["5_precision"])
    return data

def _geometric_mean(data:pd.DataFrame, column_name) -> pd.DataFrame:
    """
    Calcuates one metric score using geometric mean using precision for star 1 and 5, recall for stars 2, 3, 4

    :param data:
    :param column_name:
    :return:
    """
    data[column_name] = (data["1_precision"] +
                              data["2_recall"] +
                              data["3_recall"] +
                              data["4_recall"] +
                              data["5_precision"]) ** 1/5.
    return data


def parse_description(x: pd.Series):
    """
    Unencode information from the description of each report entry

    :param x:
    :return:
    """
    x["feature_column"] = x.description.split("-")[0]
    x["feature_engineering"] = x.description.split("-")[1]
    x["config_df"] = np.NaN if x.description.split("-")[2] == "df_none" or x.description.split("-")[2] == "df_default" else x.description.split("-")[2]
    x["config_ngram"] = np.NaN if x.description.split("-")[3] == "ngram_none" else x.description.split("-")[3]
    x["sample_size"] = x.description.split("-")[4]
    x["feature_size"] = x.description.split("-")[5]
    x["lda"] = False if x.description.split("-")[6] == "nolda" else True
    x["smote"] = False if x.description.split("-")[7] == "nosmote" else True
    x["label_column"] = x.description.split("-")[9]
    return x

def preprocess_report(report: pd.DataFrame):
    """
    This will be called by jupyter notebooks to parse our various information for a report
    :param report:
    :return:
    """
    report = calculate_metric(report)
    report = report.apply(lambda x: parse_description(), axis = 1)
    return report
