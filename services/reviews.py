#!/anaconda3/envs/capstone/bin/flask

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sys

from flask import Flask
from flask import request, abort
from flask import render_template, Blueprint

import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import sequence
import util.tf2_util as t2
import pickle
from config import Config
from logging.config import dictConfig


LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'


# TODO: load models
# MODEL_DIR = 'models'
# MODEL_JSON_FILE = f'{MODEL_DIR}/amazon_reviews_us_Wireless_v1_00-preprocessed-110k-TF2-biGRU_1layer_attention-186-star_rating-model_json.h5'
# MODEL_WEIGHTS_FILE = f'{MODEL_DIR}/amazon_reviews_us_Wireless_v1_00-preprocessed-110k-TF2-biGRU_1layer_attention-186-star_rating-weights.h5'
# TOKENIZER_FILE = f'{MODEL_DIR}/tf2-tokenizer.pkl'
model = None
tokenizer = None
MAX_FEATURES = 200


# TODO: error handling
# https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-vii-error-handling


# configure loging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(name)s.%(funcName)s[%(lineno)d] %(levelname)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


# Create the application instance
# app = Flask(__name__, template_folder="templates")
app = Flask(__name__)


app.config.from_object(Config)

# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/

    :return:        the rendered template 'home.html'
    """
    return "Hello, World!"
    #return render_template('home.html')


@app.route('/models/gru/api/v1.0/predict', methods=['POST'])
def predict_reviews():

    if not request.args or not 'review' in request.args or not 'truth' in request.args:
        abort(400)

    app.logger.debug(f'MODEL_DIR: {app.config["MODEL_DIR"]}')

    if model is None:
        load_model(f'{app.config["MODEL_DIR"]}/{app.config["MODEL_JSON_FILE"]}',
                   f'{app.config["MODEL_DIR"]}/{app.config["MODEL_WEIGHTS_FILE"]}')
    if tokenizer is None:
        load_tokenizer(f'{app.config["MODEL_DIR"]}/{app.config["TOKENIZER_FILE"]}')


    # TODO: replace this later
    text = request.args.get('review')
    truth = request.args.get('truth')
    # text = "hi there what your name"
    # truth = 2

    text_preprocessed = preprocess_text(text)
    text_encoded = encode_text([text_preprocessed])

    y = model.predict(text_encoded)

    y_unencoded = t2.unencode(y)[0]

    y_dict = convert_predictions_to_dict(y.ravel())

    app.logger.debug(f'y_dict type: { type(y_dict) }')
    app.logger.debug(f'y_dict {y_dict}')

    app.logger.debug(f'predicted y: {y}')
    json_response = render_template('response.json',
                                    status = "SUCCESS",
                                    review_raw = text,
                                    review_preprocessed = text_preprocessed,
                                    truth = truth,
                                    review_encoded = text_encoded,
                                    rating = y_unencoded,
                                    prediction_raw = json.dumps(y_dict),
                                    )
    app.logger.debug(json_response)
    app.logger.debug(f'Tensorflow version: {tf.__version__}')
    return json_response


def convert_predictions_to_dict(d: list):
    return {str(i + 1): str(d[i]) for i in range(len(d))}


def preprocess_text(text: str):
    # TODO: implement this
    return text


def encode_text(text: str):
    s = tokenizer.texts_to_sequences(text)
    app.logger.debug(f's: {s}')
    sequence_padded = sequence.pad_sequences(s,
                                 maxlen=MAX_FEATURES,
                                padding='post',
                                truncating='post')
    app.logger.debug(f'padded x: {sequence_padded}')
    return sequence_padded

def load_model(json_path: str, weights_path: str):
    """
    Load model from json configuration and weights

    :param json_path:
    :param weights_path:
    :return: model - tf.keras model
    """
    app.logger.debug("loading model")
    global model

    with open(json_path) as json_file:
        json_config = json_file.read()
    model = keras.models.model_from_json(json_config,
                                         custom_objects={'AttentionLayer': t2.AttentionLayer})
    model.load_weights(weights_path)
    return model


def load_tokenizer(tokenizer_path: str):
    """
    Loads the pre-trained tokenizer into the service

    :param tokenizer_path:
    :return:
    """
    app.logger.debug("loading tokenizer")
    global tokenizer

    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)





# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    load_model(model_json_file, model_weights_file)
    load_tokenizer(tokenizer_file)
    app.run(debug=True)