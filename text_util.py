import re
import logging
import nltk
from bs4 import BeautifulSoup
import unicodedata


logger = logging.getLogger(__name__)
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def stem_text(text: str) -> str:
    # TODO: implement this
    return text

def lemmatize_text(text: str) -> str:
    # TODO: implement this
    return text


def remove_html_tags(text: str) -> str:
    """
    Remove HTML taxs from text usig regex
    :param text: original text
    :return: stripped text
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def expand_contractions(text: str) -> str:
    # TODO: implement this
    return text


def make_lowercase(text: str) -> str:
    """
    Make text lower case
    :param text: original text
    :return: lower case string
    """
    return text.lower()


def remove_newlines(text: str) -> str:
    """
    remove newlines from text. will stirp both unix and windows
    newline characters

    :return: string without newlines
    """
    # logger.debug(f'pre-stripped: [{text}]')
    newtext = text.replace('\n', '').replace('\r', '')
    return ' '.join(newtext.split())


def remove_stop_words(text: str) -> str:
    """
    remove stop words from string
    :param text: original text
    :return: string without stop words
    """
    if text is not None and len(text) > 0:
        wpt = nltk.WordPunctTokenizer()
        tokens = wpt.tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    return text


# # Removing Accents
def remove_accented_chars(text: str) -> str:
    """
    Remove accent characters and convert to UTF-8
    :param text: original text
    :return: stripped text
    """
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# # Remove Special Characters
def remove_special_chars(text: str) -> str:
    """
    Remove anything that is not alphanumeric or spaces
    :param text:
    :return:
    """
    text = re.sub('[^a-zA-Z0-9\s]', ' ', text, re.I | re.A)
    text = remove_newlines(text)
    return ' '.join(text.split())


def remove_amazon_tags(text: str) -> str:
    """
    removes amazon tags that look like [[VIDEOID:dsfjljs]], [[ASIN:sdjfls]], etc
    :param text:
    :return:
    """
    text = re.sub('\[\[.*?\]\]', ' ', text, re.I | re.A)
    return ' '.join(text.split())

