import os

# TOOD: read these from environment variable
class Config(object):
    MODEL_DIR = 'models'
    MODEL_JSON_FILE = 'amazon_reviews_us_Wireless_v1_00-preprocessed-110k-TF2-biGRU_1layer_attention-186-star_rating-model.json'
    MODEL_WEIGHTS_FILE = 'amazon_reviews_us_Wireless_v1_00-preprocessed-110k-TF2-biGRU_1layer_attention-186-star_rating-weights.h5'
    TOKENIZER_FILE = 'tf2-tokenizer.pkl'
    PREPROCESSOR_MODULE = 'util.AmazonPreprocessor'
    PREPROCESSOR_CLASS = 'AmazonPreprocessor'
    MAX_FEATURES = 200
    SQLALCHEMY_DATABASE_URI = "sqlite:////tmp/test.db"
    VERSION = os.environ.get("VERSION", default="latest")