import pandas as pd
import numpy as np
import logging
import os

log = logging.getLogger(__name__)

EVAL_COLS=["1_recall", "2_recall", "3_recall", "4_recall", "5_precision"]

def _calculate_metric(data:pd.DataFrame, column_name ="eval_metric") -> pd.DataFrame:
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
    data[column_name] = 5 / (1/data[EVAL_COLS[0]] +
                                 1/data[EVAL_COLS[1]] +
                                 1/data[EVAL_COLS[2]] +
                                 1/data[EVAL_COLS[3]] +
                                 1/data[EVAL_COLS[4]])
    return data

def _geometric_mean(data:pd.DataFrame, column_name) -> pd.DataFrame:
    """
    Calcuates one metric score using geometric mean using precision for star 1 and 5, recall for stars 2, 3, 4

    :param data:
    :param column_name:
    :return:
    """
    data[column_name] = (data[EVAL_COLS[0]] +
                              data[EVAL_COLS[1]] +
                              data[EVAL_COLS[2]] +
                              data[EVAL_COLS[3]] +
                              data[EVAL_COLS[4]]) ** 1/5.
    return data


def _parse_description(x: pd.Series):
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
    x["has_lda"] = False if x.description.split("-")[6] == "nolda" else True
    x["lda_str"] = x.description.split("-")[6]
    x["has_sampling"] = False if x.description.split("-")[7] == "sampling_none" else True
    x["sampling_str"] = x.description.split("-")[7]
    x["label_column"] = x.description.split("-")[9]
    x["feature_summary"] = f'{x.feature_engineering}-{x.config_ngram}-{x.feature_size}'
    x["feature_summary_sampling"] = f'{x.feature_engineering}-{x.config_ngram}-{x.sampling_str}'
    return x

def _preprocess_report(report: pd.DataFrame):
    """
    This will be called by jupyter notebooks to parse our various information for a report
    :param report:
    :return:
    """
    report = _calculate_metric(report)
    report = report.apply(lambda x: _parse_description(x), axis = 1)
    return report


def load_report(filename: str):
    """
    Loads report file and preprocessed the file so we can use in our notebooks

    :param filename:
    :return:
    """
    if os.path.exists(filename):
        report = pd.read_csv(filename)
        return _preprocess_report(report)
    else:
        raise Exception(f'{filename} does not exist')


def load_best_from_report(report):
    """
    Loads the best model from a report
    :param report: report filename or data frame
    :return: Series object with the best results
    """
    if isinstance(report, str):
        report = load_report(report)
    # iloc return a Series object. Want to convert it back to a dataframe and reset the index
    # TODO: update hyperparameter notebook
    # return report.iloc[report.eval_metric.idxmax(axis=0)]
    type_dict = {idx: value for idx, value in report.dtypes.iteritems()}
    return pd.DataFrame(report.iloc[report.eval_metric.idxmax(axis=0)]).T.reset_index(drop=True).astype(type_dict, copy=False)


