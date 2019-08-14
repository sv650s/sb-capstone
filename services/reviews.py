#!/anaconda3/envs/capstone/bin/flask

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sys

from flask import Flask
from flask import render_template

import json
import logging
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import sequence
import util.tf2_util as t2
import pickle


LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)


# TODO: load models
MODEL_DIR = 'models'
model_json_file = f'{MODEL_DIR}/amazon_reviews_us_Wireless_v1_00-preprocessed-110k-TF2-biGRU_1layer_attention-186-star_rating-model_json.h5'
model_weights_file = f'{MODEL_DIR}/amazon_reviews_us_Wireless_v1_00-preprocessed-110k-TF2-biGRU_1layer_attention-186-star_rating-weights.h5'
tokenizer_file = f'{MODEL_DIR}/tf2-tokenizer.pkl'
model = None
tokenizer = None
MAX_FEATURES = 200



# response template
resp = {'prediction': 5}

# Create the application instance
# app = Flask(__name__, template_folder="templates")
app = Flask(__name__)

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


@app.route('/predict')
def predict():

    if model is None:
        load_model(model_json_file, model_weights_file)
    if tokenizer is None:
        load_tokenizer(tokenizer_file)


    # TODO: replace this later
    text = "hi there what your name"
    truth = 2

    text_preprocessed = preprocess_text(text)
    text_encoded = encode_text([text_preprocessed])

    y = model.predict(text_encoded)

    y_unencoded = t2.unencode(y)[0]

    y_dict = convert_predictions_to_dict(y.ravel())

    print(f'y_dict type: { type(y_dict) }', sys.stderr)
    print(f'y_dict {y_dict}', sys.stderr)

    print(f'predicted y: {y}', sys.stderr)
    json_response = render_template('response.json',
                                    status = "SUCCESS",
                                    review_raw = text,
                                    review_preprocessed = text_preprocessed,
                                    truth = truth,
                                    review_encoded = text_encoded,
                                    prediction = y_unencoded,
                                    prediction_raw = json.dumps(y_dict),
                                    )
    log.debug(json_response)
    print(f'Tensorflow version: {tf.__version__}', file=sys.stderr)
    return json_response


def convert_predictions_to_dict(d: list):
    return {str(i + 1): str(d[i]) for i in range(len(d))}


def preprocess_text(text: str):
    # TODO: implement this
    return text


def encode_text(text: str):
    s = tokenizer.texts_to_sequences(text)
    print(f's: {s}', sys.stderr)
    sequence_padded = sequence.pad_sequences(s,
                                 maxlen=MAX_FEATURES,
                                padding='post',
                                truncating='post')
    print(f'padded x: {sequence_padded}', sys.stderr)
    return sequence_padded

def load_model(json_path: str, weights_path: str):
    """
    Load model from json configuration and weights

    :param json_path:
    :param weights_path:
    :return: model - tf.keras model
    """
    print("loading model", sys.stderr)
    global model

    with open(json_path) as json_file:
        json_config = json_file.read()
    model = keras.models.model_from_json(json_config,
                                         custom_objects={'AttentionLayer': t2.AttentionLayer})
    model.load_weights(model_weights_file)
    return model


def load_tokenizer(tokenizer_path: str):
    """
    Loads the pre-trained tokenizer into the service

    :param tokenizer_path:
    :return:
    """
    print("loading tokenizer", sys.stderr)
    global tokenizer

    with open(tokenizer_file, 'rb') as file:
        tokenizer = pickle.load(file)





# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    load_model(model_json_file, model_weights_file)
    load_tokenizer(tokenizer_file)
    app.run(debug=True)