
from util.AmazonTextNormalizer import AmazonTextNormalizer
from tensorflow.keras.preprocessing import sequence
import logging

logger = logging.getLogger(__name__)

class Preprocessor(object):

    def __init__(self):
        self.tokenizer = None
        self.normalizer = None

    def preprocess(self, text: str):
        raise Exception("not yet implemented")


class TokenizedPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__()
        self.max_features = None
        self.tokenizer = None
        self.normalizer = AmazonTextNormalizer()


    def pad_text(self, text: str):
        """
        Pad text to have max number of features
        :param text:
        :return:
        """
        text_sequence = self.tokenizer.texts_to_sequences(text)
        logger.debug(f's: {text_sequence}')
        sequence_padded = sequence.pad_sequences(text_sequence,
                                                 maxlen=self.max_features,
                                                 padding='post',
                                                 truncating='post')
        logger.debug(f'padded x: {sequence_padded}')
        return text_sequence, sequence_padded


    def preprocess(self, text: str):

        logger.debug(f"Preprocessing text [{text}]")
        text_normalized = self.normalizer.normalize_text(text)
        logger.debug(f"Preprocessed text [{text_normalized}]")

        text_sequence, text_encoded = self.pad_text([text_normalized])
        logger.debug(f"Encoded text [{text_normalized}]")

        return text_normalized, text_sequence, text_encoded



