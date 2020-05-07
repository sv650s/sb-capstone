import pandas as pd
import numpy as np
import logging
import os
import json
import util.keras_util as ku
import util.dict_util as du

log = logging.getLogger(__name__)

EVAL_COLS=["1_recall", "2_recall", "3_recall", "4_recall", "5_precision"]

def calculate_metric(data, column_name ="eval_metric", dnn = False) -> pd.DataFrame:
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
        log.info("Calculating metric for ML report")
        data[column_name] = data.apply(lambda x: _harmonic_mean(
            [ x[col] for col in EVAL_COLS ] ), axis=1)
    elif isinstance(data, dict):
        log.info("calculating metric from dictionary")
        log.debug(f'{data}')
        m = []
        m.append(data["1"]["recall"])
        m.append(data["2"]["recall"])
        m.append(data["3"]["recall"])
        m.append(data["4"]["recall"])
        m.append(data["5"]["precision"])
        log.info("got all values to calculate")

        return _harmonic_mean(m)
    return data

def calculate_metric15(data, column_name ="eval_metric", dnn = False) -> pd.DataFrame:
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
        log.info("Calculating metric for ML report")
        data[column_name] = data.apply(lambda x: _harmonic_mean(
            [ x[col] for col in EVAL_COLS ] ), axis=1)
    elif isinstance(data, dict):
        log.info("calculating metric from dictionary")
        log.debug(f'{data}')
        m = []
        m.append(data["1"]["recall"])
        m.append(data["2"]["precision"])
        log.info("got all values to calculate")

        return _harmonic_mean(m)
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
            mean = 0
            break
        else:
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
    x["config_df"] = np.NaN if x.description.split("-")[2] == "df_none" or x.description.split("-")[2] == "df_none" else x.description.split("-")[2]
    x["config_ngram"] = np.NaN if x.description.split("-")[3] == "ngram_none" else x.description.split("-")[3]
    x["sample_size"] = x.description.split("-")[4]
    x["feature_size"] = x.description.split("-")[5]
    x["has_lda"] = False if x.description.split("-")[6] == "nolda" else True
    x["lda_str"] = x.description.split("-")[6]
    x["has_sampling"] = False if x.description.split("-")[7] == "sampling_none" else True
    x["sampling_type"] = x.description.split("-")[7]
    x["label_column"] = x.description.split("-")[9]
    x["feature_summary"] = f'{x.feature_engineering}-{x.config_ngram}-{x.feature_size}'
    x["feature_summary_sampling"] = f'{x.feature_engineering}-{x.config_ngram}-{x.sampling_type}'
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


def _preprocess_dnn_report_file(report: pd.DataFrame):
    """
    preprocess dnn report and parse out extra information that wasn't in the original report

    columns it will create:
        eval_metric - this will be calculated by calculate_metric function
        sample_size = training_examples + test_examples
        display_name = model_name + architecture


    :param report:
    :return:
    """
    # TODO: implement parsing out other attributes
    report["eval_metric"] = report["classification_report"].apply(lambda x: calculate_metric(json.loads(x)))
    report["sample_size"] = report.train_examples + report.test_examples
    if "architecture" in report.keys():
        report["display_name"] = report["model_name"] + " (" + report["architecture"] + ")"
    else:
        report["display_name"] = report["model_name"]
    return report

def convert_dnn_report_format(report:pd.DataFrame):
    """
    DNN reports have a slighlty different format. Classification report is stored as a json instead of flattened in the report
    We will convert the dnn format to standard format here
    :param report:
    :return:
    """
    new_report = pd.DataFrame()
    for idx, row in report.iterrows():
        d = row.to_dict()
        cr_dict = json.loads(row.classification_report)
        d = du.add_dict_to_dict(d, cr_dict)
        new_report = new_report.append(d, ignore_index=True, sort=False)
    return new_report

def load_dnn_report(report_dir: str, report_file = None, convert_format = False):
    """
    DNN report has a slightly different format

    Use this function to load the file and pre-process it
    :param report_dir: directory where report is stored
    :param report_file: filename of report
    :param convert_format: convert it to standard format that ML models used. Default False
    :return: report dataframe
    """
    if report_file is None:
        report_file = ku.ModelWrapper.get_report_file_name(report_dir)
    else:
        report_file = f'{report_dir}/{report_file}'
    assert os.path.exists(report_file), f"report file missing {report_file}"
    report = pd.read_csv(report_file, quotechar="'")
    if convert_format:
        report = convert_dnn_report_format(report)
    report = _preprocess_dnn_report_file(report)
    return report



