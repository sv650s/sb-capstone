import os

# TOOD: read these from environment variable
class Config(object):
    LOCAL_MODEL_DIR = 'models'
    MODEL_CACHE_DIR = '/tmp'
    MAX_FEATURES = 200
    SQLALCHEMY_DATABASE_URI = "sqlite:////tmp/test.db"
    VERSION = os.environ.get("VERSION", default="latest")
    PROJECT_ID = os.environ.get("PROJECT_ID")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    DEFAULT_MODEL_CONFIG = "config/GRU-v1.0.json"
    MODEL_CONFIG_DIR = "config"
    MODEL_LOADER_MODULE = os.environ.get("MODEL_LOADER_MODULE", default="util.model_util")
    MODEL_LOADER_CLASS = os.environ.get("MODEL_LOADER_CLASS", default="LocalModelLoader")
