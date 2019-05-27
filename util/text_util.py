import re
import logging
import nltk
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from .contraction_map import CONTRACTION_MAP
import unicodedata
import sys
import traceback

# set up logger
logger = logging.getLogger(__name__)

# global variables
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
FILE_DATE_FORMAT = '%Y-%m-%d-%H'


def remove_stop_words_from_list(words: list):
    for word in words:
        if word in stop_words:
            logger.info(f"Removing the following from stop words: {word}")
            stop_words.remove(word)


def stem_text(text: str) -> str:
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(ps.stem(word))
    return ' '.join(stemmed_words)

def lemmatize_text(text: str) -> str:
    # TODO: implement this - currently points to stem_words
    assert False, "Lemmatization is not yet implemented"
    return stem_text(text)


def remove_html_tags(text: str) -> str:
    """
    Remove HTML taxs from text usig regex
    :param text: original text
    :return: stripped text
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def expand_contractions(text: str, contraction_mapping=CONTRACTION_MAP) -> str:

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


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
        try:
            tokens = wpt.tokenize(text)
            filtered_tokens = [word for word in tokens if word not in stop_words]
            return ' '.join(filtered_tokens)
        except:
            traceback.print_exc()
            logger.error(f"got error trying to tokenize this string [{text}]")
            sys.exit(1)
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
    text = re.sub('[^a-zA-Z0-9\s]', ' ', text, flags=re.I | re.A)
    text = remove_newlines(text)
    return ' '.join(text.split())


def get_contractions(text: str) -> list:
    """
    returns a list of contractions from text
    :param text:
    :return:
    """
    text = text.replace('\n',' ').replace('\r', ' ').replace('\t', ' ').replace('’', "'")
    logger.debug(f"looking for contractions [{text}]")
    contraction_list = []
    # this doesn't capture if the word is at the end of the line
    for (match) in re.findall(r"\s+([a-z]+'[a-z]{1,2})[\s\t\n]+", text.lower(), flags=re.ASCII | re.IGNORECASE):
        contraction_list.append(match)
    # match if contraction is at the end of the line
    for (match) in re.findall(r"\s+([a-z]+'[a-z]{1,2})$", text.lower(), flags=re.ASCII | re.IGNORECASE):
        contraction_list.append(match)

    logger.debug(f"found the following contractons [{contraction_list}]")
    return contraction_list


def remove_alphanumeric_words(x):
    """
    In amazon reviews especially for wireless categories you have some alpha numeric word
    which represent model numbers, we want to remove this

    Will also remove words that are basically only numbers
    :param x:
    :return:
    """
    # remove mixed words
    x = re.sub(r'\s*([a-z]+[\d]+[\w\d]*|[\d]+[a-z]+[\d\w]*|[\d]{2,})', '', x, flags=re.IGNORECASE)
    # remove numbers
    # x = re.sub(r'\s(\d+)', '', x)
    return ' '.join(x.split())
