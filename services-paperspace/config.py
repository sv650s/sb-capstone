import os


# TOOD: read these from environment variable
class Config(object):

    # location to load models from
    MODEL_DIR = '/models' # location where models are saved
    # maximum number of features inputed into model - ie, padded sequence
    MAX_FEATURES = 100
    # Type of database to use. Option: mysql, sqlite
    DB_TYPE = os.environ.get("DB_TYPE", default="mysql")
    # DB_TYPE = os.environ.get("DB_TYPE", default="sqlite")
    # Connection string for DB
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://reviews:password@{os.environ.get("DB_IP")}/capstonedb'
    # SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/capstone.db'
    # Flask service version
    VERSION = os.environ.get("VERSION", default="1") # default model version
    # env variable will be set by docker-compose file - default is for local development
    MODEL_BUILDER_CLASS = os.environ.get("MODEL_BUILDER_CLASS", default="util.model_builder.PaperspaceLocalModelBuilder")
    # location to store SQLite DB if DB_TYPE == sqlite
    SQLITE_DIR = os.environ.get("SQLITE_DIR", default="/tmp")
    # how to load model. Possible values: json, SavedModel, h5
    MODEL_FORMAT = os.environ.get("MODEL_FORMAT", default="json")
