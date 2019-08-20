#
# This is a factory class that abstracts how we load the model. It is implemented as a singleton so it should
# be thread safe
#
from flask import current_app as app
import tensorflow.keras as keras
import pickle
import util.tf2_util as t2
import importlib
from tensorflow.keras.preprocessing import sequence
import logging

logger = logging.getLogger(__name__)


class Classifier(object):
    """
    Encapsulated model class that abstracts away pre-processing and inference from the user
    """

    @staticmethod
    def get_key(name: str, version: str):
        return f'{name}_{version}'

    # TODO: make feature_encoder required
    def __init__(self, name: str,
                 version: str,
                 model, tokenizer, preprocessor, max_features, feature_encoder=None, label_encoder=t2):
        """

        :param name:
        :param version:
        :param model:
        :param tokenizer:
        :param preprocessor:
        :param max_features:
        :param feature_encoder:
        :param label_encoder:
        """
        self.name = name
        self.version = version
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.feature_encoder = feature_encoder
        self.label_encoder = label_encoder
        self.max_features = max_features

    def __str__(self):
        return Classifier.get_key(self.name, self.version)

    def encode_text(self, text: str):
        s = self.tokenizer.texts_to_sequences(text)
        logger.debug(f's: {s}')
        sequence_padded = sequence.pad_sequences(s,
                                                 maxlen=self.max_features,
                                                 padding='post',
                                                 truncating='post')
        logger.debug(f'padded x: {sequence_padded}')
        return sequence_padded

    def predict(self, text):
        # TODO: implement later
        logger.debug(f"Preprocessing text [{text}]")
        text_preprocessed = self.preprocessor.normalize_text(text)
        logger.debug(f"Preprocessed text [{text_preprocessed}]")

        # TODO: figure refactor out feaure encoder
        text_encoded = self.encode_text([text_preprocessed])
        # text_encoded = self.feature_encoder.encode(text_preprocessed)
        logger.debug(f"Encoded text [{text_preprocessed}]")

        y_raw = self.model.predict(text_encoded)
        logger.debug(f"y_raw [{y_raw}]")

        y_unencoded = self.label_encoder.unencode(y_raw)[0]
        logger.debug(f"y_unencoded [{y_unencoded}]")

        return y_unencoded, y_raw, text_preprocessed, text_encoded


class ModelStore(object):
    """
    Wrapper to store various models
    Underlying structure is a dictionary
    Models will be referenced by name and version
    """

    def __init__(self):
        self.model_map = {}

    # def put(self, model_name: str,
    #         model_version: str,
    #         model: Classifier):
    #     logger.debug(f'adding {model_name}_{model_version} to store {model}')
    def put(self, model):
        self.model_map[str(model)] = model

    def get(self,
            model_name: str,
            model_version: str):
        """
        Get model associated with the name and version
        :param model_name:
        :param model_version:
        :return:
        """
        return self.model_map[Classifier.get_key(model_name, model_version)]


class FileModelFactory(object):
    # map that stores all models and associated files
    _model_cache = ModelStore()

    @staticmethod
    def get_model(name: str, version='latest'):
        """
        Checks to see if it's in the cache, if not, try to load it
        :param name:
        :param version:
        :return:
        """
        model = None
        try:
            model = FileModelFactory._model_cache.get(name, version)
            logger.info(f"got {Classifier.get_key(name, version)} from cache")
        except Exception as e:
            logger.info(str(e))
        finally:
            if not model:
                logger.info(f"{Classifier.get_key(name, version)} not in cache. loading...")
                model = FileModelFactory._load_model(name, version)
                logger.debug(f'Finished loading model {model}')
                if model:
                    FileModelFactory._model_cache.put(model)

        return model

    @staticmethod
    def _load_model(name: str, version: str):
        """
        Load model from json configuration and weights

        :param json_path:
        :param weights_path:
        :return: model - tf.keras model
        """
        model_json = f'{app.config["MODEL_DIR"]}/{app.config["MODEL_JSON_FILE"]}'
        model_weights_json = f'{app.config["MODEL_DIR"]}/{app.config["MODEL_WEIGHTS_FILE"]}'

        logger.info(f"loading model from {model_json}")
        with open(model_json) as json_file:
            json_config = json_file.read()
        model = keras.models.model_from_json(json_config,
                                             custom_objects={'AttentionLayer': t2.AttentionLayer})
        logger.info(f"loading model weights from {model_weights_json}")
        model.load_weights(model_weights_json)

        tokenizer_path = f'{app.config["MODEL_DIR"]}/{app.config["TOKENIZER_FILE"]}'
        logger.info(f"loading tokenizer from {tokenizer_path}")
        with open(tokenizer_path, 'rb') as file:
            tokenizer = pickle.load(file)

        preprocessor_module = app.config["PREPROCESSOR_MODULE"]
        preprocessor_class = app.config["PREPROCESSOR_CLASS"]
        logger.info(f"loading preprocessor from {preprocessor_module}.{preprocessor_class}")
        # pmodule = __import__(preprocessor_module)
        # pclass_ = getattr(pmodule, preprocessor_class)
        pclass_ = getattr(importlib.import_module(preprocessor_module), preprocessor_class)
        preprocessor = pclass_()

        classifier = Classifier(name, version, model, tokenizer, preprocessor, app.config['MAX_FEATURES'])

        return classifier
