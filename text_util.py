import re
import logging
import nltk
import unicodedata


logger = logging.getLogger(__name__)
stop_words = nltk.corpus.stopwords.words('english')


wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def stem_text(text: str) -> str:
    # TODO: implement this
    return text


def expand_contractions(text: str) -> str:
    # TODO: implement this
    return text

# # make text lower case


def make_lowercase(text: str) -> str:
    """
    make text lower case
    """
    return text.lower()


def remove_stop_words(text: str) -> str:
    """
    remove stop words from string


    Returns
    -------
    returns string without stop words
    """
    if text is not None and len(text) > 0:
        wpt = nltk.WordPunctTokenizer()
        tokens = wpt.tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    return text


# # Removing Accents
def remove_accented_chars(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# # Remove Special Characters
def remove_special_chars(text: str) -> str:
    """
    remove anything that is not characters or numbers
    """
    text = re.sub('[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    return text.replace('\n', '').replace('\r', '')
