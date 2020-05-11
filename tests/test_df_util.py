import pytest
import pandas as pd
import numpy as np
import util.df_util as dfu
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class TestDfUtil(object):


    @pytest.fixture(scope="class")
    def int_df(self):
        """
        set up a matrix with numbers
        :return:
        """
        a = {"a": [0, 1, 2], "b": [3, 4, 5], "c": [6, 7, 8]}
        return pd.DataFrame(a)


    @pytest.fixture(scope="class")
    def float_df(self):
        """
        set up a matrix with numbers
        :return:
        """
        a = {"a": [0., 0.1, 0.2], "b": [0.3, 0.4, 0.5], "c": [0.6, 0.7, 0.8]}
        return pd.DataFrame(a)

    @pytest.fixture(scope="class")
    def class_column_df(self, class_name):
        a = {class_name: [10,11,12]}
        return pd.DataFrame(a)


    @pytest.fixture(scope="class")
    def class_name(self):
        return "class"


    def test_cast_smaller_type(self, int_df, float_df, class_column_df, class_name):
        # test no casting
        in_df = int_df.join(class_column_df)
        out_df = dfu.cast_samller_type(in_df, class_name)
        dtypes = out_df.dtypes
        assert dtypes.get("a") == np.dtype("int64"), "a should be int64"
        assert dtypes.get("b") == np.dtype("int64"), "b should be int64"
        assert dtypes.get("c") == np.dtype("int64"), "c should be int64"
        assert dtypes.get(class_name) == np.dtype("int64"), "a should be int64"



        # test casting int8
        in_df = int_df.join(class_column_df)
        out_df = dfu.cast_samller_type(in_df, class_name, "int8")
        dtypes = out_df.dtypes
        assert dtypes.get("a") == np.dtype("int8"), "a should be int8"
        assert dtypes.get("b") == np.dtype("int8"), "b should be int8"
        assert dtypes.get("c") == np.dtype("int8"), "c should be int8"
        assert dtypes.get(class_name) == np.dtype("int8"), f"{class_name} should be int8"

        # test casting uint32
        in_df = int_df.join(class_column_df)
        out_df = dfu.cast_samller_type(in_df, class_name, "uint32")
        dtypes = out_df.dtypes
        assert dtypes.get("a") == np.dtype("uint32"), "a should be uint32"
        assert dtypes.get("b") == np.dtype("uint32"), "b should be uint32"
        assert dtypes.get("c") == np.dtype("uint32"), "c should be uint32"
        assert dtypes.get(class_name) == np.dtype("int8"), f"{class_name} should be int8"


        # test casting float32
        in_df = float_df.join(class_column_df)
        out_df = dfu.cast_samller_type(in_df, class_name, "float32")
        dtypes = out_df.dtypes
        assert dtypes.get("a") == np.dtype("float32"), "a should be float32"
        assert dtypes.get("b") == np.dtype("float32"), "b should be float32"
        assert dtypes.get("c") == np.dtype("float32"), "c should be float32"
        assert dtypes.get(class_name) == np.dtype("int8"), f"{class_name} should be int8"


    def test_cast_column_type(self, int_df, float_df):
        # test no casting
        out_df = dfu.cast_column_type(int_df, "c")
        dtypes = out_df.dtypes
        assert dtypes.get("a") == np.dtype("int64"), "a should be int64"
        assert dtypes.get("b") == np.dtype("int64"), "b should be int64"
        assert dtypes.get("c") == np.dtype("int64"), "c should be int64"


        # test cast column c
        out_df = dfu.cast_column_type(int_df, "c", "int8")
        dtypes = out_df.dtypes
        assert dtypes.get("a") == np.dtype("int64"), "a should be int64"
        assert dtypes.get("b") == np.dtype("int64"), "b should be int64"
        assert dtypes.get("c") == np.dtype("int8"), "c should be int8"


    def test_drop_empty_columns(self):
        # test that we are dropping rows with 0 len strings
        # test that we are dropping rows with NaN column values
        df = pd.DataFrame({ "a": [ "nonempty", "", np.NaN, "nonempty", "nonempty" ],
                            "b": [ "nonempty", "nonempty", "nonempty", "", np.NaN ]})
        new_df = dfu.drop_empty_columns(df, ["a", "b"])
        assert len(new_df) == 1, f"Len should be 1 after dropping rows: {new_df}"



