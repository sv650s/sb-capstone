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
from os import path
import json
from pprint import pprint
import util.gcp_file_util as gu
import util.python_util as pu

# TODO: figure out why logging doens't work without app.app.logger
# logger = logging.getLogger(__name__)



class ModelLoader(object):

    def load_model(self, model_file: str, weights_file: str = None, custom_objects = None):
        pass

    def load_tokenizer(self, tokenizer_file: str):
        pass

    def get_config(self, filename):
        pass

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


class LocalModelLoader(ModelLoader):
    # TODO: make loaders ignositic of flask app - somehow pass in the model dir dynamically

    def load_tokenizer(self, tokenizer_file: str):
        app.logger.info(f"loading tokenizer from {tokenizer_file}")
        with open(f'{app.config["LOCAL_MODEL_DIR"]}/{tokenizer_file}', 'rb') as file:
            tokenizer = pickle.load(file)
        return tokenizer

    def load_model(self, model_file: str, weights_file: str = None, custom_objects = None):
        app.logger.info(f"loading model from {model_file}")
        with open(f'{app.config["LOCAL_MODEL_DIR"]}/{model_file}') as json_file:
            json_config = json_file.read()
        model = keras.models.model_from_json(json_config,
                                             custom_objects=custom_objects)
        app.logger.info(f"loading model weights from {weights_file}")
        model.load_weights(f'{app.config["LOCAL_MODEL_DIR"]}/{weights_file}')

        return model

    def get_config(self, filename):
        json_file = f'{app.config["MODEL_CONFIG_DIR"]}/{filename}'
        app.logger.debug(f"config json_file {json_file}")
        return json_file


class GCPModelLoader(ModelLoader):

    def __init__(self):
        super()
        self.model_dir = app.config['MODEL_CACHE_DIR']
        self.bucket_name = app.config['BUCKET_NAME']

    def _download_file(self, filename):
        app.logger.info(f'downloaded {filename} to {self.model_dir}')
        store_path = f'{self.model_dir}/{filename}'
        gu.download_blob(self.bucket_name,
                         filename,
                         store_path)
        return store_path


    def load_model(self, model_file: str, weights_file: str, custom_objects: dict = None):

        self._download_file(model_file)
        app.logger.info(f"loading model from {model_file}")
        with open(f'{self.model_dir}/{model_file}') as json_file:
            json_config = json_file.read()
        model = keras.models.model_from_json(json_config,
                                             custom_objects=custom_objects)

        self._download_file(weights_file)
        app.logger.info(f"loading model weights from {weights_file}")
        model.load_weights(f'{self.model_dir}/{weights_file}')

        return model

    def load_tokenizer(self, tokenizer_file: str):
        self._download_file(tokenizer_file)
        app.logger.info(f"loading tokenizer from {tokenizer_file}")
        with open(f'{self.model_dir}/{tokenizer_file}', 'rb') as file:
            tokenizer = pickle.load(file)
        return tokenizer

    def get_config(self, filename):
        return self._download_file(filename)


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

    # TODO: dynamically load padding as part of preprocessing
    def pad_text(self, text: str):
        s = self.tokenizer.texts_to_sequences(text)
        app.logger.debug(f's: {s}')
        sequence_padded = sequence.pad_sequences(s,
                                                 maxlen=self.max_features,
                                                 padding='post',
                                                 truncating='post')
        app.logger.debug(f'padded x: {sequence_padded}')
        return sequence_padded

    @staticmethod
    def get_config_filename(name: str, version: str):
        filename = f'{name}-{version}.json'
        app.logger.debug(f"got filename: {filename}")
        return filename

    @staticmethod
    def from_json(json_file: str, model_loader: ModelLoader):
        """
        Recreates a model based on json configuration

        :param json_file: filepath for the json configuration
        :return: model
        """

        classifier = None
        if path.exists(json_file):

            with open(json_file, 'r') as file:
                config_str = file.read()

            config = json.loads(config_str)

            app.logger.debug(f"json_config {pprint(config)}")

            name = config["name"]
            version = config["version"]

            model_json = config["model"]
            model_weights_json = config["weights"]
            model = model_loader.load_model(model_json, model_weights_json, ModelLoader.get_custom_objects(config["custom_objects"]))

            tokenizer_path = config["tokenizer"]
            tokenizer = model_loader.load_tokenizer(tokenizer_path)

            preprocessor = pu.load_instance(config['preprocessor'])

            classifier = Classifier(name, version, model, tokenizer, preprocessor, app.config['MAX_FEATURES'])

        return classifier

    def predict(self, text):
        """
        does the follwing:
        1. preprocess text
        2. padding
        3. calls predict on the model
        :param text:
        :return:
        """
        app.logger.debug(f"Preprocessing text [{text}]")
        text_preprocessed = self.preprocessor.normalize_text(text)
        app.logger.debug(f"Preprocessed text [{text_preprocessed}]")

        # TODO: figure refactor out feaure encoder
        text_encoded = self.pad_text([text_preprocessed])
        # text_encoded = self.feature_encoder.encode(text_preprocessed)
        app.logger.debug(f"Encoded text [{text_preprocessed}]")

        y_raw = self.model.predict(text_encoded)
        app.logger.debug(f"y_raw [{y_raw}]")

        y_unencoded = self.label_encoder.unencode(y_raw)[0]
        app.logger.debug(f"y_unencoded [{y_unencoded}]")

        return y_unencoded, y_raw, text_preprocessed, text_encoded


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

                # dynamically create loader based configuration
                loader_classpath = app.config["MODEL_LOADER_CLASS"]
                app.logger.info(f"Creating loader from {loader_classpath}")
                loader = pu.load_instance(loader_classpath)
                app.logger.debug(f"created loader {loader}")

                json_file = loader.get_config(Classifier.get_config_filename(name, version))
                app.logger.debug(f"model_config_json {json_file}")

                model = Classifier.from_json(json_file, loader)
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
