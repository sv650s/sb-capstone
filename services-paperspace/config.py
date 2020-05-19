import os


# TOOD: read these from environment variable
class Config(object):
    LOCAL_MODEL_DIR = '/models'
    MODEL_CACHE_DIR = '/tmp' # this is where GCPModelBuilder will store configuration files
    MAX_FEATURES = 100
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://reviews:password@{os.environ.get("DB_IP")}/capstonedb'
    VERSION = os.environ.get("VERSION", default="1") # default model version
    DEFAULT_MODEL_CONFIG = "models/GRU-v1.0.json"
    MODEL_CONFIG_DIR = "models"
    # env variable will be set by docker-compose file - default is for local development
    MODEL_BUILDER_CLASS = os.environ.get("MODEL_BUILDER_CLASS", default="util.model_builder.PaperspaceLocalModelBuilder")
    DB_TYPE = os.environ.get("DB_TYPE", default="sqlite")
    SQLITE_DIR = os.environ.get("SQLITE_DIR", default="/tmp")
