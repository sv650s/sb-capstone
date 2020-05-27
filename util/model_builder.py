#
# This is a factory class that abstracts how we load the model. It is implemented as a singleton so it should
# be thread safe
#
from abc import ABC, abstractmethod
import logging
from flask import current_app as app
import tensorflow.keras as keras
import pickle
import os
import json
from pprint import pprint
import util.python_util as pu
import util.tf2_util as t2
import util.service_preprocessor as sp


logger = logging.getLogger(__name__)

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
        logger.debug(f"got filename: {filename}")
        return filename

    def build(self):
        json_file = self.get_json_config_filepath()
        classifier = None

        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                config_str = file.read()
                logger.debug(f'config_str {config_str}')

            config = json.loads(config_str)

            logger.debug(f"json_config {pprint(config)}")

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


# TODO: uncomment and merge this with PaperspaceLocalModelBuilder
# class GCPLocalModelBuilder(ModelBuilder):
#
#     # TODO: make loaders ignositic of flask app - somehow pass in the model dir dynamically
#     def __init__(self, name, version):
#         super().__init__(name, version)
#         self.model_dir = app.config["LOCAL_MODEL_DIR"]
#         self.config_dir = app.config["MODEL_CONFIG_DIR"]
#
#     def load_tokenizer(self, tokenizer_file: str):
#         logger.info(f"loading tokenizer from {tokenizer_file}")
#         with open(f'{self.model_dir}/{tokenizer_file}', 'rb') as file:
#             tokenizer = pickle.load(file)
#         return tokenizer
#
#     def load_model(self, model_file: str, weights_file: str = None, custom_objects=None):
#         logger.info(f"loading model from {model_file}")
#         with open(f'{self.model_dir}/{model_file}') as json_file:
#             json_config = json_file.read()
#         model = keras.models.model_from_json(json_config,
#                                              custom_objects=custom_objects)
#         logger.info(f"loading model weights from {weights_file}")
#         model.load_weights(f'{self.model_dir}/{weights_file}')
#
#         return model
#
#     def load_encoder(self, encoder: str):
#         pass
#
#     def get_json_config_filepath(self):
#         json_file = f'{self.config_dir}/{self.get_config_filename()}'
#         logger.debug(f"config json_file {json_file}")
#         return json_file


# TODO: extend from ModelBuilder
class PaperspaceLocalModelBuilder(object):
    """
    Loads a model in paperspace machine VM and returns a model object when build() is called
    """
    def __init__(self, model_name, version, model_dir):
        """

        :param model_name:  name of model to load
        :param version: version of model to load
        :param model_dir: directory where model files are saved
        """
        self.model_name = model_name
        self.version = version
        self.model_dir = model_dir

    def build(self, max_features, load_format):
        """
        Load the model from file
        :param max_features: max number of features per sample
        :param load_format: format to load from. Possible values: json, SavedModel, h5
        :return:
        """
        # TODO: remove this
        print(f'__name__: {__name__}')
        logger.info(f"loading model from format: {load_format}")
        model = None

        model_name_with_version = f'{self.model_name}-v{self.version}'
        model_path = f'{self.model_dir}/{model_name_with_version}'
        model_h5_file = f'{model_path}/{model_name_with_version}-model.h5'
        model_json_file = f'{model_path}/{model_name_with_version}-model.json'
        weights_file = f'{model_path}/{model_name_with_version}-weights.h5'
        logger.info(f"Atempting to load model from: {model_path}")

        if os.path.exists(model_path):
            if load_format == "SavedModel":
                logger.info("loading model from SavedModel format")
                # TODO: loading from SavedModel doesn't work because of architecture
                # model = keras.models.load_model(f'{model_path}/{self.version}')
                raise NotImplemented("Loading SavedModel not yet implemented")
                pass

            elif load_format == "json":
                logger.info("loading model from json format")
                # load from JSON file
                logger.info(f"loading model from {model_json_file}")
                with open(model_json_file) as json_file:
                    json_config = json_file.read()
                model = keras.models.model_from_json(json_config)
                logger.info(f"loading model weights from {weights_file}")
                model.load_weights(weights_file)


            elif load_format == "h5":
                logger.info("loading model from h5 format")
                # TODO: loading h5 doesn't seem to work
                # logger.info(f'Loading model from: {model_h5_file}')
                # model = keras.models.load_model(model_h5_file)
                raise NotImplemented("Loading h5 model not yet implemented")
                pass

            else:
                raise Exception(f"Unknown format for loading: {load_format}")


        else:
            raise FileNotFoundError(model_path)

        # load tokenizer
        tokenizer_file = f'{model_path}/{self.model_name}-v{self.version}-tokenizer.pkl'
        logger.info(f"loading tokenizer from {tokenizer_file}")
        with open(tokenizer_file, 'rb') as file:
            tokenizer = pickle.load(file)


        preprocessor = sp.TokenizedPreprocessor()
        preprocessor.tokenizer = tokenizer
        preprocessor.max_features = max_features

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
        return f'{name}-v{version}'

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
        :param label_encoder: fn to Unencodes the raw results back to numeric classes. Must implement unencode function
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
        logger.debug(f"text_preprocessed [{text_preprocessed}]")
        logger.debug(f"text_sequence [{text_sequence}]")
        logger.debug(f"sequence_padded [{sequence_padded}]")
        logger.debug(f"reversed sequence [{self.preprocessor.tokenizer.sequences_to_texts(text_sequence)}]")

        y_raw = self.model.predict(sequence_padded)
        logger.debug(f"y_raw [{y_raw}]")

        # dynamically load unencoder
        y_unencoded = self.label_encoder.unencode(y_raw)[0]
        logger.debug(f"y_unencoded [{y_unencoded}]")

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
    Factory class to load and store models
    """
    _instance = None
    _model_location = None
    _max_features = None
    _load_format = None

    @staticmethod
    def init_factory(model_location:str, max_features:int, load_format:str):
        """
        Initialize the factory so it knows how to load models

        :param model_location: filepath to models
        :param max_features: max features per sample (ie, padded sequence)
        :param load_format: format to load model from - possible values: json, h5
        :return:
        """
        logger.info(f'Initializing ModelFactory with:\n' \
                    f'\tmodel_location: {model_location}' \
                        f'\tmax_features: {max_features}' \
                        f'\tload_format: {load_format}' )
        if not os.path.exists(model_location):
            raise FileNotFoundError(model_location)
        ModelFactory._model_location = model_location
        ModelFactory._max_features = max_features
        ModelFactory._load_format = load_format


    @staticmethod
    def get_instance():
        if ModelFactory._instance is None:
            ModelFactory._instance = ModelFactory(ModelFactory._model_location,
                                                  ModelFactory._max_features,
                                                  ModelFactory._load_format)
        return ModelFactory._instance


    def __init__(self,
                 model_location,
                 max_features = 100,
                 load_format = "json"):
        """

        :param model_location: location to load model from
        :param max_features: maximum number of features in a sample. Default 100
        :param load_format: format to load models from. possible values: json, SavedModel, h5. Default json
        """
        self._model_location = model_location
        self.max_features = max_features
        self.load_format = load_format
        # map that stores all models and associated files
        self._model_cache = ModelCache()


    def get_model(self, name: str, version):
        """
        Checks to see if it's in the cache, if not, try to load it
        :param name:
        :param version:
        :return:
        """
        logger.info(f"Getting model name: {name} version: {version}")
        model = None
        try:
            model = self._model_cache.get(name, version)
            logger.info(f"got {Classifier.get_key(name, version)} from cache")
        except Exception as e:
            logger.info(str(e))
        finally:
            if not model:
                # TOOD: figure out how to change behavior using some type of inheritance
                logger.info(f"{Classifier.get_key(name, version)} not in cache. loading...")


                # TODO: dynamically generate this class based on config
                # dynamically create builder based configuration
                # builder_classpath = app.config["MODEL_BUILDER_CLASS"]
                # logger.info(f"Creating builder from {builder_classpath}")
                # builder = pu.load_instance(builder_classpath, name, version)
                # TODO: change PaperspaceLocalModelBuilder to a factory - factory should be
                # saved as an instance in ModelFactory and will be reused to build all models
                # ModelBuilder can be instantiated in the main flask app to be more clear
                builder = PaperspaceLocalModelBuilder(name, version, self._model_location)
                logger.debug(f"created builder {builder}")

                model = builder.build(self.max_features, self.load_format)
                logger.debug(f'Finished loading model {model}')
                if model is not None:
                    self._model_cache.put(model)

        return model

    def put(self, model: Classifier):
        self._model_cache.put(model)

    def clear(self):
        logger.info("clearning model cache")
        self._model_cache.clear()
        logger.info(f"cache size after clearing :{self._model_cache.size()}")

    def model_list(self):
        """
        returns a list of keys stored in the cache
        :return:
        """
        return list(self._model_cache.keys())
