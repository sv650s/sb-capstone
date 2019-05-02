import pandas as pd
from pandas import DataFrame
from pandas import Series
import logging
import text_util as tu

logger = logging.getLogger(__name__)
# counter for debugging
counter = 0


class TextPreprocessor:

    def __init__(self,
                 text_columns,
                 columns_to_drop=None,
                 to_lowercase=True,
                 remove_newlines=True,
                 remove_amazon_tags=True,
                 remove_html_tags=True,
                 remove_accented_chars=True,
                 expand_contractions=True,
                 remove_special_chars=True,
                 stem_text=True,
                 lemmatize_text=True,
                 remove_stop_words=True):
        """
        :param text_columns: list of column names with text you want to process. required
        :param columns_to_drop: list of column names to drop
        :param to_lowercase:
        :param remove_newlines:
        :param remove_amazon_tags:
        :param remove_html_tags:
        :param remove_accented_chars:
        :param expand_contractions:
        :param remove_special_chars:
        :param stem_text:
        :param lemmatize_text:
        :param remove_stop_words:
        """
        assert text_columns is not None, "text_column_name is required"

        self.text_columns = text_columns
        self.columns_to_drop = columns_to_drop
        self.to_lowercase = to_lowercase
        self.remove_newlines = remove_newlines
        self.remove_amazon_tags = remove_amazon_tags
        self.remove_html_tags = remove_html_tags
        self.remove_accented_chars = remove_accented_chars
        self.expand_contractions = expand_contractions
        self.remove_special_chars = remove_special_chars
        self.stem_text = stem_text
        self.lemmatize_text = lemmatize_text
        self.remove_stop_words = remove_stop_words

    def drop_columns(self, df: DataFrame) -> DataFrame:
        """
        Drops columns specified in columns_to_drop from constructor
        :param df: original DataFrame
        :return: DataFrame without the columns specified
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
            logger.info(f'normalizing [{counter}] row: [{row}]')
            counter += 1

        for column in self.text_columns:
            text = row[column]
            # text = row.text
            # logger.debug(row.describe())
            logger.debug(f'column: [{column}] text: [{text}]')

            if text is not None and len(text) > 0:

                # use regex to make lower case
                if self.to_lowercase:
                    text = tu.make_lowercase(text)
                if self.remove_newlines:
                    text = tu.remove_newlines(text)
                if self.remove_amazon_tags:
                    text = tu.remove_amazon_tags(text)
                if self.remove_html_tags:
                    text = tu.remove_html_tags(text)
                if self.remove_accented_chars:
                    text = tu.remove_accented_chars(text)
                if self.expand_contractions:
                    text = tu.expand_contractions(text)
                if self.remove_special_chars:
                    text = tu.remove_special_chars(text)
                if self.stem_text:
                    text = tu.stem_text(text)
                if self.lemmatize_text:
                    text = tu.lemmatize_text(text)
                if self.remove_stop_words:
                    text = tu.remove_stop_words(text)

            logger.debug(f'clean text from column [{column}] [{text}]')
            row[column] = text
        return row

    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """
        Do the following:
            * strip columns
            * normalize text
        """
        logger.info("start preprocessing data")
        if self.columns_to_drop:
            df = self.drop_columns(df)

        # reset counter
        global counter
        counter = 0
        df = df.apply(self.normalize_text, axis=1)

        logger.info("finished preprocessing data")
        logger.info(df.info())
        logger.info(df.head())

        return df

    def convert_to_csv(self, df: DataFrame, outfile: str):
        """
        convert df to cvs
        """
        df[self.text_column_name].apply(tu.remove_newlines)
        df.to_csv(outfile, index=False, doublequote=True)
