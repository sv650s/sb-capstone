import pandas as pd
from pandas import DataFrame
from pandas import Series
import logging
import nltk
import text_util as tu
import numpy as np

logger = logging.getLogger(__name__)
# counter for debugging
counter = 0


def strip_review_columns(df: DataFrame) -> DataFrame:
    """
        retain only review_id, text, and stars columns
    """
    drop_columns = ['cool', 'date', 'funny', 'useful', 'user_id']
    return df.drop(drop_columns, axis=1)


def normalize_row(row: Series) -> Series:
    """
    Noramlize text
    1. make lower case
    2. remove accents
    3. remove special characters
    """
    assert row is not None, "row is None"

    global counter

    if counter % 1000 == 0:
        logger.debug(f'normalizing [{counter}] row: [{row}]')
        counter += 1

    text = row.text

    # use regex to make lower case
    text = tu.make_lowercase(text)
    text = tu.remove_accented_chars(text)
    # TODO: expand contractions
    text = tu.remove_special_chars(text)
    text = tu.stem_text(text)
    text = tu.remove_stop_words(text)

    row.text = text
    return row


def remove_newline_from_row(row: DataFrame) -> DataFrame:
    """
    Utility function to remove newlines
    """
    row.text = tu.remove_special_chars(row.text)
    return row


def process_data(df: DataFrame) -> DataFrame:
    """
    Do the following:
        * strip columns
        * normalize text
    """
    df = strip_review_columns(df)

    # reset counter
    counter = 0
    df = df.apply(normalize_row, axis=1)

    logger.debug(df.head())

    return df


def convert_to_csv(df: DataFrame, outfile: str):
    """
    convert df to cvs
    """
    df = df.apply(remove_newline_from_row, axis=1)
    df.to_csv(outfile, index=False, doublequote=True)
