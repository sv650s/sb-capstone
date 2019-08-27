import os


# TOOD: read these from environment variable
class Config(object):
    LOCAL_MODEL_DIR = 'models'
    MODEL_CACHE_DIR = '/tmp' # this is where GCPModelBuilder will store configuration files
    MAX_FEATURES = 200
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://root:freel00k@{os.environ.get("DB_IP")}/capstonedb?' \
        'ssl_key=credentials/client-key.pem&ssl_cert=credentials/client-cert.pem'
    VERSION = os.environ.get("VERSION", default="latest")
    PROJECT_ID = os.environ.get("PROJECT_ID")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    DEFAULT_MODEL_CONFIG = "models/GRU-v1.0.json"
    MODEL_CONFIG_DIR = "models"
    # env variable will be set by docker-compose file - default is for local development
    MODEL_BUILDER_CLASS = os.environ.get("MODEL_BUILDER_CLASS", default="util.model_util.LocalModelBuilder")
