#
# This is a factory class that abstracts how we load the model. It is implemented as a singleton so it should
# be thread safe
#
from abc import ABC, abstractmethod
import logging
from flask import current_app as app
import tensorflow.keras as keras
import pickle
from os import path
import json
from pprint import pprint
import util.python_util as pu
import util.tf2_util as t2
import util.service_preprocessor as sp


# TODO: figure out why logging doens't work without app.app.logger
logger = logging.getLogger(__name__)
# app.logger = logging.getLogger(__name__)


class ModelBuilder(ABC):

    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version

    @abstractmethod
    def load_model(self, model_file: str, weights_file: str = None, custom_objects=None):
        pass

    @abstractmethod
    def load_tokenizer(self, tokenizer_file: str):
        pass

    @abstractmethod
    def load_encoder(self, encoder: str):
        pass

    @abstractmethod
    def get_json_config_filepath(self):
        pass

    def get_config_filename(self):
        """
        Based on name and version, get the config filename without path
        :return:
        """
        filename = f'{self.model_name}-{self.version}.json'
        app.logger.debug(f"got filename: {filename}")
        return filename

    def build(self):
        json_file = self.get_json_config_filepath()
        classifier = None

        if path.exists(json_file):
            with open(json_file, 'r') as file:
                config_str = file.read()
                app.logger.debug(f'config_str {config_str}')

            config = json.loads(config_str)

            app.logger.debug(f"json_config {pprint(config)}")

            name = config["name"]
            version = config["version"]

            model_json = config["model"]
            model_weights_json = config["weights"]
            model = self.load_model(model_json,
                                    model_weights_json,
                                    ModelBuilder.get_custom_objects(config["custom_objects"]))

            tokenizer_path = config["tokenizer"]
            tokenizer = self.load_tokenizer(tokenizer_path)

            preprocessor = pu.load_instance(config['preprocessor'])
            preprocessor.tokenizer = tokenizer
            preprocessor.max_features = app.config['MAX_FEATURES']

            classifier = Classifier(name, version, model, preprocessor)

        return classifier

    @staticmethod
    def get_custom_objects(d: dict):
        """
        Converts a dictionary of string to custom object used for load_model function

        Values in the dictionary will be dynamically converted into the class representation
        """
        ret_d = {}
        for k, v in d.items():
            ret_d[k] = pu.load_class(v)
        return ret_d


# class GCPLocalModelBuilder(ModelBuilder):
#
#     # TODO: make loaders ignositic of flask app - somehow pass in the model dir dynamically
#     def __init__(self, name, version):
#         super().__init__(name, version)
#         self.model_dir = app.config["LOCAL_MODEL_DIR"]
#         self.config_dir = app.config["MODEL_CONFIG_DIR"]
#
#     def load_tokenizer(self, tokenizer_file: str):
#         app.logger.info(f"loading tokenizer from {tokenizer_file}")
#         with open(f'{self.model_dir}/{tokenizer_file}', 'rb') as file:
#             tokenizer = pickle.load(file)
#         return tokenizer
#
#     def load_model(self, model_file: str, weights_file: str = None, custom_objects=None):
#         app.logger.info(f"loading model from {model_file}")
#         with open(f'{self.model_dir}/{model_file}') as json_file:
#             json_config = json_file.read()
#         model = keras.models.model_from_json(json_config,
#                                              custom_objects=custom_objects)
#         app.logger.info(f"loading model weights from {weights_file}")
#         model.load_weights(f'{self.model_dir}/{weights_file}')
#
#         return model
#
#     def load_encoder(self, encoder: str):
#         pass
#
#     def get_json_config_filepath(self):
#         json_file = f'{self.config_dir}/{self.get_config_filename()}'
#         app.logger.debug(f"config json_file {json_file}")
#         return json_file


class PaperspaceLocalModelBuilder(object):
    """

    """
    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version

    def build(self):
        model = None

        file_dir = app.config["LOCAL_MODEL_DIR"]

        model_path = f'{file_dir}/{self.model_name}'
        model_file = f'{model_path}/{self.model_name}-model.json'
        weights_file = f'{model_path}/{self.model_name}-weights.h5'
        app.logger.info(f"Atempting to load model from: {model_path}")

        if path.exists(model_path):
            # model = keras.models.load_model(f'{model_path}/{self.version}')

            app.logger.info(f"loading model from {model_file}")
            with open(model_file) as json_file:
                json_config = json_file.read()
            model = keras.models.model_from_json(json_config)
            app.logger.info(f"loading model weights from {weights_file}")
            model.load_weights(weights_file)
        else:
            raise FileNotFoundError(model_path)

        # load tokenizer
        tokenizer_file = f'{model_path}/{self.model_name}-tokenizer.pkl'
        app.logger.info(f"loading tokenizer from {tokenizer_file}")
        with open(tokenizer_file, 'rb') as file:
            tokenizer = pickle.load(file)


        preprocessor = sp.TokenizedPreprocessor()
        preprocessor.tokenizer = tokenizer
        preprocessor.max_features = app.config['MAX_FEATURES']

        classifier = Classifier(self.model_name, self.version, model, preprocessor)

        return classifier



class Classifier(object):
    """
    Encapsulated model class that abstracts away pre-processing and inference from the user

    __str_ method has been modified - this will return an identifier for the model using name and version - you can do
        this either by converting the model into a string or using the get_key method
    """

    @staticmethod
    def get_key(name: str, version: str):
        return f'{name}_{version}'

    # TODO: make feature_encoder required
    def __init__(self, name: str,
                 version: str,
                 model,
                 preprocessor,
                 label_encoder=t2):
        """

        :param name:
        :param version:
        :param model:
        :param tokenizer:
        :param preprocessor:
        :param max_features:
        :param feature_encoder:
        :param label_encoder: must implement unencode function
        """
        self.name = name
        self.version = version
        self.model = model
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.max_features = None

    def __str__(self):
        return Classifier.get_key(self.name, self.version)

    def predict(self, text):
        """
        does the follwing:
        1. preprocess text
        2. padding
        3. calls predict on the model
        :param text:
        :return:
        """
        text_preprocessed, text_sequence, sequence_padded = self.preprocessor.preprocess(text)

        y_raw = self.model.predict(sequence_padded)
        app.logger.debug(f"y_raw [{y_raw}]")

        # dynamically load unencoder
        y_unencoded = self.label_encoder.unencode(y_raw)[0]
        app.logger.debug(f"y_unencoded [{y_unencoded}]")

        return y_unencoded, y_raw, text_preprocessed, text_sequence


class ModelCache(object):
    """
    Dictionary wrapper to store various models
    Underlying structure is a dictionary
    Models will be referenced by name and version
    """

    def __init__(self):
        self.model_map = {}

    def put(self, model):
        """
        Stores model
        :param model:
        :return:
        """
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

    def size(self):
        """
        return number of items in the cache
        :return:
        """
        return len(self.model_map)

    def keys(self):
        """
        returns list of keys in our cache
        :return:
        """
        return list(self.model_map.keys())

    def clear(self):
        """
        clears the cache
        :return:
        """
        self.model_map.clear()


class ModelFactory(object):
    """
    Use this class to reconstruct the model
    """

    # map that stores all models and associated files
    _model_cache = ModelCache()

    @staticmethod
    def get_model(name: str, version='latest'):
        """
        Checks to see if it's in the cache, if not, try to load it
        :param name:
        :param version:
        :return:
        """
        app.logger.info(f"Getting model name: {name} version: {version}")
        model = None
        try:
            model = ModelFactory._model_cache.get(name, version)
            app.logger.info(f"got {Classifier.get_key(name, version)} from cache")
        except Exception as e:
            app.logger.info(str(e))
        finally:
            if not model:
                # TOOD: figure out how to change behavior using some type of inheritance
                app.logger.info(f"{Classifier.get_key(name, version)} not in cache. loading...")

                # dynamically create builder based configuration
                builder_classpath = app.config["MODEL_BUILDER_CLASS"]
                app.logger.info(f"Creating builder from {builder_classpath}")

                # TODO: dynamically generate this class based on config
                # builder = pu.load_instance(builder_classpath, name, version)
                builder = PaperspaceLocalModelBuilder(name, version)
                app.logger.debug(f"created builder {builder}")

                model = builder.build()
                app.logger.debug(f'Finished loading model {model}')
                if model:
                    ModelFactory._model_cache.put(model)

        return model

    @staticmethod
    def put(model: Classifier):
        ModelFactory._model_cache.put(model)

    @staticmethod
    def clear():
        app.logger.info("clearning model cache")
        ModelFactory._model_cache.clear()
        app.logger.info(f"cache size after clearing :{ModelFactory._model_cache.size()}")

    @staticmethod
    def model_list():
        """
        returns a list of keys stored in the cache
        :return:
        """
        return list(ModelFactory._model_cache.keys())
