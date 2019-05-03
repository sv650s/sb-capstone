import re
import logging
import nltk
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import unicodedata

# set up logger
logger = logging.getLogger(__name__)

# global variables
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
stop_words.remove('do')
stop_words.remove('did')
stop_words.remove('does')
stop_words.remove('very')
ps = PorterStemmer()


def stem_text(text: str) -> str:
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(ps.stem(word))
    return ' '.join(stemmed_words)

def lemmatize_text(text: str) -> str:
    # TODO: implement this - currently points to stem_words
    assert False, "Not yet implemented"
    return stem_text(text)


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


