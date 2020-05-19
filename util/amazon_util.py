import logging
import util.text_util as tu
import re

logger = logging.getLogger(__name__)
# counter for debugging

# these words are meaningful in reviews - we will not remove these stop words
# when pre-processing our text
STOP_WORDS_TO_REMOVE=[
    'no',
    'not',
    'do',
    'don',
    "don't",
    'does',
    'did',
    'does',
    'doesn',
    "doesn't",
    'should',
    'very',
    'will'
]


def remove_amazon_tags(text: str) -> str:
    """
    removes amazon tags that look like [[VIDEOID:dsfjljs]], [[ASIN:sdjfls]], etc
    :param text:
    :return:
    """
    logger.debug(f"before amazon tags {text}")
    text = re.sub(r'\[\[.*?\]\]', ' ', text, re.I | re.A)
    text = ' '.join(text.split())
    logger.debug(f"after processing amazon tags {text}")
    return text

def post_processor_replace_numbers_with_words(text: str) -> str:
    """
    replace 1 to 5 with one to five so we can capture things like 5 stars
    :param text:
    :return:
    """
    # TODO: implement this
    # text = re.sub(r'\[\[.*?\]\]', ' ', text, re.I | re.A)
    return text

def remove_http_links(text: str) -> str:
    """
    Amazon reviews sometimes have http tags that link to images. Want to remove these
    :param text:
    :return:
    """
    text = re.sub(r'(http[s]{0,1}:\S+)', '', text, re.I | re.A)
    return text



class AmazonTextNormalizer:

    def __init__(self,
                 stop_word_remove_list=STOP_WORDS_TO_REMOVE,
                 to_lowercase=True,
                 remove_newlines=True,
                 remove_html_tags=True,
                 remove_accented_chars=True,
                 expand_contractions=True,
                 remove_special_chars=True,
                 stem_text=False,
                 lemmatize_text=False,
                 remove_alphanumeric_words=True,
                 remove_stop_words=False):
        """

        :param text_columns: list of columns to process. required
        :param columns_to_drop: list of columns to drop. optional
        :param stop_word_remove_list: list of stopwords to remove
        :param to_lowercase:
        :param remove_newlines:
        :param remove_html_tags:
        :param remove_accented_chars:
        :param expand_contractions:
        :param remove_special_chars:
        :param stem_text:  default = False
        :param lemmatize_text: default = False
        :param remove_stop_words:
        :param retain_original_columns:
        :param custom_preprocessor: pass it a custom function to run before any processing. this can be a function or a list
        :param custom_postprocessor: pass it a custom function to run after processing
        """
        self.stop_word_remove_list = stop_word_remove_list
        self.to_lowercase = to_lowercase
        self.remove_newlines = remove_newlines
        self.remove_html_tags = remove_html_tags
        self.remove_accented_chars = remove_accented_chars
        self.expand_contractions = expand_contractions
        self.remove_special_chars = remove_special_chars
        self.stem_text = stem_text
        # we are either going to stem or lemmatize
        self.lemmatize_text = lemmatize_text
        self.remove_alphanumeric_words = remove_alphanumeric_words
        self.remove_stop_words = remove_stop_words

        if self.stop_word_remove_list:
            tu.remove_stop_words_from_list(self.stop_word_remove_list)


    def normalize_text(self, text: str) -> str:
        """
        Noramlize text
        1. make lower case
        2. remove accents
        3. remove special characters
        """
        assert text is not None, "row is None"

            # text = row.text
            # logger.debug(row.describe())
        logger.info(f'normalizing text: {text}')

        if text is not None and len(text) > 0:

            text = remove_amazon_tags(text)
            text = remove_http_links(text)

            if self.to_lowercase:
                text = tu.make_lowercase(text)
            if self.remove_newlines:
                text = tu.remove_newlines(text)
            if self.remove_html_tags:
                text = tu.remove_html_tags(text)
            if self.remove_accented_chars:
                text = tu.remove_accented_chars(text)
            if self.expand_contractions:
                text = tu.expand_contractions(text)
            if self.remove_special_chars:
                text = tu.remove_special_chars(text)
            # we have to do this after expanding contractions so it doesn't remove words like don't or shouldn't
            if self.remove_alphanumeric_words:
                text = tu.remove_alphanumeric_words(text)
            if self.remove_stop_words:
                text = tu.remove_stop_words(text)
            if self.stem_text:
                text = tu.stem_text(text)
            if self.lemmatize_text:
                text = tu.lemmatize_text(text)

            text = self.expand_star_ratings(text)

        logger.info(f'finished normalizing text: {text}')
        return text


