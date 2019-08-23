import os


# TOOD: read these from environment variable
class Config(object):
    LOCAL_MODEL_DIR = 'models'
    MODEL_CACHE_DIR = '/tmp'
    MAX_FEATURES = 200
    # TODO: get SSL working
    # SQLALCHEMY_DATABASE_URI = f'mysql://root@{os.environ.get("DB_IP")}/capstone-db?' \
    #     'ssl_key=credentials/client-key.pem&ssl_cert=credentials/client-cert.pem'
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://root:freel00k@{os.environ.get("DB_IP")}/capstonedb'
    VERSION = os.environ.get("VERSION", default="latest")
    PROJECT_ID = os.environ.get("PROJECT_ID")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    DEFAULT_MODEL_CONFIG = "config/GRU-v1.0.json"
    MODEL_CONFIG_DIR = "config"
    MODEL_LOADER_MODULE = os.environ.get("MODEL_LOADER_MODULE", default="util.model_util")
    MODEL_LOADER_CLASS = os.environ.get("MODEL_LOADER_CLASS", default="LocalModelLoader")
