from google.cloud import storage
import logging
from flask import current_app
import util.model_builder as mb
from flask import current_app as app
import tensorflow.keras as keras
import pickle

logger = logging.getLogger(__name__)


def _get_storage_client():
    return storage.Client(
        project=current_app.config['PROJECT_ID'])



def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    logger.info('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


class GCPModelBuilder(mb.ModelBuilder):
    """
    Extends ModelBuilder to load model from GCP bucket instead of local file system
    """

    def __init__(self, name, version):
        super().__init__(name, version)
        self.model_dir = app.config['MODEL_CACHE_DIR']
        self.bucket_name = app.config['BUCKET_NAME']

    def _download_file(self, filename):
        """
        download file from gcp to local dir then return the filepath of the config file
        :param filename:
        :return:
        """
        app.logger.info(f'downloaded {filename} to {self.model_dir}')
        store_path = f'{self.model_dir}/{filename}'
        download_blob(self.bucket_name,
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

    def load_encoder(self, encoder: str):
        pass

    def get_json_config_filepath(self):
        return self._download_file(self.get_config_filename())