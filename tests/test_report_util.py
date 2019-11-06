import pytest
import pandas as pd
import numpy as np
import logging
import util.report_util as ru

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def test_load_dnn_report(shared_datadir):
    """
    Test processing and loading of DNN report and make sure it's being pre-processed
    :return:
    """

    report = ru.load_dnn_report(shared_datadir)
    assert isinstance(report, pd.DataFrame), f'report is not a dataframe: {type(report)}'
    assert "eval_metric" in report.columns, f'eval_metric missing in report columns: {report.columns}'
    assert report.loc[0, "eval_metric"] == 5/(1/0.5 + 1/0.1 + 1/0.3 + 1/0.3 + 1/0.8 ), f"eval_metric calculation incorrect {report.loc[0, 'eval_metric']}"
