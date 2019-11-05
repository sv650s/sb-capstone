import pandas as pd
import numpy as np
import logging
import os

log = logging.getLogger(__name__)

EVAL_COLS=["1_recall", "2_recall", "3_recall", "4_recall", "5_precision"]

def calculate_metric(data:pd.DataFrame, column_name ="eval_metric", dnn = False) -> pd.DataFrame:
    """
    Calculates the harmonic mean in the following manner so we can use one metric to evalute our models
        recall - star rating 1, 2, 3, 4
        precision - star rating 5

    :param data: if it's a report df - must have 1_precision, 2_recall, 3_recall, 4_recall, and 5_precision columns.
        Or you can pass in a classification dictionary
    :param column_name: name of column to put metric in. default eval_metric
    :param dnn: indicates if we are calculating this for a DNN notebook as reports have different format
    :return:
    """
    if isinstance(data, pd.DataFrame):
        if dnn:
            log.info("Calculating metric for dnn report")
            data[column_name] = data.classification_report.apply(lambda x:
                                                                 _harmonic_mean([
                                                                     x["1"]["recall"],
                                                                     x["2"]["recall"],
                                                                     x["3"]["recall"],
                                                                     x["4"]["recall"],
                                                                     x["5"]["precision"]
                                                                 ]))

        else:
            log.info("Calculating metric for ML report")
            data[column_name] = 5 / (1/data[EVAL_COLS[0]] +
                                     1/data[EVAL_COLS[1]] +
                                     1/data[EVAL_COLS[2]] +
                                     1/data[EVAL_COLS[3]] +
                                     1/data[EVAL_COLS[4]])
    else:
        log.info("calculating metric from dictionary")
        m1 = data["1"]["recall"]
        m2 = data["2"]["recall"]
        m3 = data["3"]["recall"]
        m4 = data["4"]["recall"]
        m5 = data["5"]["precision"]

        if m1 > 0 and m2 > 0 and m3 > 0 and m4 > 0 and m5 > 0:
            data = 5 / (1 / m1 +
                        1 / m2 +
                        1 / m3 +
                        1 / m4 +
                        1 / m5)
        else:
            data = 0
    return data

def _harmonic_mean(values: list):
    """
    Calculates the harmonic mean based on a list of values

    if any of the items in the list is 0, function will return 0

    :param values:
    :return:
    """
    mean = 0
    for v in values:
        if v == 0:
            break
        mean += 1 / v
    if mean > 0:
        mean = len(values) / mean

    return mean




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
    report = calculate_metric(report)
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
    type_dict = {idx: value for idx, value in report.dtypes.iteritems()}
    return pd.DataFrame(report.iloc[report.eval_metric.idxmax(axis=0)]).T.reset_index(drop=True).astype(type_dict, copy=False)


