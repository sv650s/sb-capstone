import pandas as pd
import numpy as np
import logging
from pprint import pformat

log = logging.getLogger(__name__)

def duplicate_columns(df:pd.DataFrame, columns:list, reorder:bool = True) -> pd.DataFrame:
    """
    duplicates specified columns and add a _orig to the column name
    :param columns: columns in DF to duplicate
    :param reorder: if specified, it will put the column and _orig together so it's easier to see. Default = True
    :return:
    """
    for column in columns:
        df[f'{column}_orig'] = df[column]
        new_column_list = []
    for column in list(df.columns.values):
        if '_orig' not in column:
            if column in columns:
                # add column and the orig
                new_column_list.append(f'{column}_orig')
                new_column_list.append(column)
            else:
                # just add the column
                new_column_list.append(column)
    df = df[new_column_list]
    return df


def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    drop columns from dataframe
    :param df:  df to drop from
    :param columns_to_drop: list of column names
    :return: new df with columns dropped
    """
    return df.drop(columns_to_drop, axis=1)


def drop_empty_columns(df: pd.DataFrame, columns_to_check) -> pd.DataFrame:
    """
    drop any rows where there are empty values in columns specified in columns_to_checks
    :param df:
    :param columns_to_check:
    :return:
    """
    # drop null columns
    df.dropna(subset = columns_to_check, inplace = True)
    # drop non-null columns that have 0 length
    for column in columns_to_check:
        df = df[
            df[column].apply(lambda x: len(x) > 0)
        ]
    return df




def cast_samller_type(df: pd.DataFrame, class_column: str, new_type: str=None) -> pd.DataFrame:
    """
    feature files are sparse matrices which doesn't need the memory space of the default np.int64 or np.float64
    we are going to cast it down to smaller types so it doesn't take up as much memory
    :param df:
    :return:
    """
    if new_type:
        log.info(f"Casting down to {new_type}")
        dtype_dict = {a: np.dtype(new_type) for a in df.columns if a not in [class_column]}
        log.debug(pformat(dtype_dict))
        dtype_dict[class_column] = np.int8
        return df.astype(dtype_dict, copy=False)
    return df


def cast_column_type(df: pd.DataFrame, column_name: str, new_type: str=None) -> pd.DataFrame:
    """
    cast a specific column in df to specific type
    :param df:
    :param column_name:
    :param new_type:
    :return:
    """
    if new_type:
        log.info(f"Casting column {column_name} to {new_type}")
        dtype_dict = {column_name: np.dtype(new_type)}
        log.debug(pformat(dtype_dict))
        return df.astype(dtype_dict, copy=False)
    return df
