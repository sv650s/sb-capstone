#!/anaconda3/envs/capstone/bin/flask

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sys

from flask import Flask
from flask import request, abort
from flask import render_template, Blueprint
from flask_sqlalchemy import SQLAlchemy

from datetime import datetime
import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import sequence
import util.tf2_util as t2
import pickle
from config import Config
from logging.config import dictConfig
from util.AmazonPreprocessor import AmazonPreprocessor

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
db = SQLAlchemy(app)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_raw = db.Column(db.String(255), nullable=False)
    input_preprocessed = db.Column(db.String(255), nullable=True)
    input_encoded = db.Column(db.String(500), nullable=True)
    class_expected = db.Column(db.Integer, nullable=False)
    class_predicted = db.Column(db.Integer, nullable=False)
    class_predicted_raw = db.Column(db.String(100), nullable=False)  # json of softmax output
    model_version = db.Column(db.String(10), nullable=False)
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    status = db.Column(db.String(32), nullable=False)

    def __repr__(self):
        return f'<ID: {self.id}\tEXPECTED: {self.class_expected}\tPREDICTED: {self.class_predicted}'

    def to_json(self):
        return render_template('response.json',
                               status=self.status,
                               review_raw=self.input_raw,
                               review_preprocessed=self.input_preprocessed,
                               review_encoded=self.input_encoded,
                               truth=self.class_expected,
                               rating=self.class_predicted,
                               prediction_raw=self.class_predicted_raw)

    def __str__(self):
        return self.to_json()


app.logger.info("creating database...")
db.create_all()


# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/

    :return:        the rendered template 'home.html'
    """
    return f"Welcome to Capstone Service - Version {app.config['VERSION']}!\n"


@app.route('/models/api/v1.0/gru', methods=['POST'])
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

    app.logger.debug(f'y_dict type: {type(y_dict)}')
    app.logger.debug(f'y_dict {y_dict}')

    app.logger.debug(f'predicted y: {y}')
    json_response = render_template('response.json',
                                    status="SUCCESS",
                                    review_raw=text,
                                    review_preprocessed=text_preprocessed,
                                    truth=truth,
                                    review_encoded=json.dumps(text_encoded.ravel().tolist()),
                                    rating=y_unencoded,
                                    prediction_raw=json.dumps(y_dict))
    app.logger.debug(json_response)
    app.logger.debug(f'Tensorflow version: {tf.__version__}')

    # save response to DB
    history = PredictionHistory(input_raw=text,
                                input_preprocessed=text_preprocessed,
                                input_encoded=json.dumps(text_encoded.ravel().tolist()),
                                class_expected=truth,
                                class_predicted=y_unencoded,
                                class_predicted_raw=json.dumps(y_dict),
                                model_version="v1.0",  # get this from the URL
                                status="SUCCESS")

    db.session.add(history)
    db.session.commit()

    return json_response


@app.route('/history/api/v1.0', methods=['GET'])
def get_history():
    # TODO: change to handle multiple
    results = PredictionHistory.query.all()
    # resp = "[ "
    # for result in results:
    #     resp += result.to_json()
    #
    # resp += " ]"

    app.logger.debug(type(results))
    resp = '[ ' + ", ".join(str(result) for result in results) + " ]"


    # return results[0].to_json()
    return resp

def convert_predictions_to_dict(d: list):
    return {str(i + 1): str(d[i]) for i in range(len(d))}


def preprocess_text(text: str):
    # TODO: implement this

    ap = AmazonPreprocessor()
    return ap.normalize_text(text)


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
    app.logger.info("loaidng model...")
    load_model(f'{app.config["MODEL_DIR"]}/{app.config["MODEL_JSON_FILE"]}',
               f'{app.config["MODEL_DIR"]}/{app.config["MODEL_WEIGHTS_FILE"]}')
    app.logger.info("loaidng tokenizer...")
    load_tokenizer(f'{app.config["MODEL_DIR"]}/{app.config["TOKENIZER_FILE"]}')
    app.run(host='0.0.0.0', debug=True)
