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


class TextPreprocessor:

    def __init__(self,
                 text_column_name,
                 columns_to_drop=None,
                 to_lowercase=False,
                 remove_newlines=False,
                 remove_html_tags=True,
                 remove_accented_chars=False,
                 expand_contractions=False,
                 remove_special_chars=False,
                 stem_text=False,
                 remove_stop_words=False):
        """
        Constructor
            text_column_name - column where we want text we want to normalize is
        """
        assert text_column_name is not None, "text_column_name is required"

        self.text_column_name = text_column_name
        self.columns_to_drop = columns_to_drop
        self.to_lowercase = to_lowercase
        self.remove_newlines = remove_newlines
        self.remove_html_tags = remove_newlines
        self.remove_accented_chars = remove_accented_chars
        self.expand_contractions = expand_contractions
        self.remove_special_chars = remove_special_chars
        self.stem_text = stem_text
        self.remove_stop_words = remove_stop_words

    def drop_columns(self, df: DataFrame) -> DataFrame:
        """
            drop columns specifed in drop_columns
        """
        return df.drop(self.columns_to_drop, axis=1)

    def normalize_text(self, row: Series) -> Series:
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

        text = row[self.text_column_name]
        # text = row.text
        # logger.debug(row.describe())
        # logger.debug(f'text from row [{text}]')

        # use regex to make lower case
        if self.to_lowercase:
            text = tu.make_lowercase(text)
        if self.remove_newlines:
            text = tu.remove_newlines(text)
        if self.remove_html_tags:
            text = tu.remove_html_tags(text)
        if self.remove_accented_chars:
            text = tu.remove_accented_chars(text)
        if self.expand_contractions:
            # TODO: expand contractions
            pass
        if self.remove_special_chars:
            text = tu.remove_special_chars(text)
        if self.stem_text:
            text = tu.stem_text(text)
        if self.remove_stop_words:
            text = tu.remove_stop_words(text)

        row[self.text_column_name] = text
        return row

    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """
        Do the following:
            * strip columns
            * normalize text
        """
        logger.debug("start preprocessing data")
        if self.columns_to_drop:
            df = self.drop_columns(df)

        # reset counter
        counter = 0
        df = df.apply(self.normalize_text, axis=1)

        logger.debug("finished preprocessing data")
        logger.debug(df.head())

        return df

    def convert_to_csv(self, df: DataFrame, outfile: str):
        """
        convert df to cvs
        """
        df[self.text_column_name] = tu.remove_newlines
        df.apply(tu.remove_newlines, axis=1)
        df.to_csv(outfile, index=False, doublequote=True)
