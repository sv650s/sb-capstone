import pandas as pd



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
    for column in columns_to_check:
        df = df[
            df[column].apply(lambda x: len(x) > 0)
        ]
    return df