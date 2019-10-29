import pytest
import pandas as pd
from tools import run_classifiers as rc
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class TestRunClassifiers(object):

    @pytest.fixture("class")
    def in_df(self):
        d = {"a": [1, 1, 1], "b": [2, 2, 2]}
        return pd.DataFrame(d)




    def test_create_training_data(self, in_df):
        x_train, x_test, y_train, y_test = rc.create_training_data(in_df, "b")

        print(x_train)
        print(y_train)
        assert len(x_train.columns) == 1, "x has too many columns"
        assert x_train.columns[0] == "a", "x should only have column a"
        assert y_train.iloc[0] == 2, "y should have 2's"




