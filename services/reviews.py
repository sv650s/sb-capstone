#!/anaconda3/envs/capstone/bin/flask

from __future__ import absolute_import, division, print_function, unicode_literals

from flask import Flask
from flask import request, abort
from flask import render_template, Blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_restplus import Api, Resource, reqparse, fields

from datetime import datetime
import json
import tensorflow as tf
from config import Config
from util.model_util import ModelFactory
from logging.config import dictConfig

TIMESTAMP = "%Y-%m-%d %H:%M:%S"

# TODO: error handling
# https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-vii-error-handling


# configure loging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] [%(levelname)s] %(name)s.%(funcName)s[%(lineno)d]: %(message)s',
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
api = Api(app, version=app.config['VERSION'], title='Vince Capstone Service',
          description='Classification model for reviews')
db = SQLAlchemy(app)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_raw = db.Column(db.String(255), nullable=False)
    input_preprocessed = db.Column(db.String(255), nullable=True)
    input_encoded = db.Column(db.String(1024), nullable=True)
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
db.session.commit()
app.logger.info("finished creating database...")


def get_factory():
    # return getattr(importlib.import_module(app.config['MODEL_FACTORY_MODULE']), app.config['MODEL_FACTORY_CLASS'])
    return ModelFactory


def get_cache_response():
    model_names = get_factory().model_list()
    json_reponse = render_template('response-cache.json',
                                   status="SUCCESS",
                                   timestamp=datetime.now().strftime(TIMESTAMP),
                                   size=len(model_names),
                                   model_names=json.dumps(model_names))
    return json_reponse


# Create a URL route in our application for "/"
# @api.route('/')
# def home():
#     """
#     This function just responds to the browser ULR
#     localhost:5000/
#
#     :return:        the rendered template 'home.html'
#     """
#     return f"Welcome to Capstone Service - Version {app.config['VERSION']}!\n"

@api.route('/model/cache', endpoint='model')
class ModelCache(Resource):

    def delete(self):
        return clear_models(), 200

    def get(self):
        return cache(), 200




@api.route('/model/<string:model>/<string:version>')
@api.doc(params={'model': 'model name - ie, GRU',
                 'version': 'model version - ie, v1.0'})
class ModelPrediction(Resource):

    # input model for predict function
    predict_model = api.model("predict", {
        "review": fields.String(title="Review", required=True),
        "product_rating": fields.Integer(title="class rating - 1 to 5")
    })

    @api.expect(predict_model)
    def post(self, model, version):
        input = api.payload
        app.logger.debug(f'input {input}')
        return predict_reviews(model, version, input['review'], input['product_rating']), 200


@api.route('/prediction/history')
class ModelHistory(Resource):

    def get(self):
        return get_history(), 200


# @api.route('/models/api/v1.0/clear_cache', methods=['PUT'])
def clear_models():
    """
    Reset and clear all cached models
    :return:
    """
    get_factory().clear()
    return get_cache_response()


# @api.route('/models/api/v1.0/cache', methods=['GET'])
def cache():
    """
    Reset and clear all cached models
    :return:
    """
    model_names = get_factory().model_list()
    return get_cache_response()


# @api.route('/models/api/v1.0/gru', methods=['POST'])
def predict_reviews(model, version, text, truth):
    # if not request.args or not 'review' in request.args or not 'truth' in request.args:
    #     abort(400)
    app.logger.info(f'Tensorflow version: {tf.__version__}')

    # TODO: un-hard code this - get this from the URL
    classifier = get_factory().get_model(model, version)
    if classifier:
        y_unencoded, y_raw, text_preprocessed, text_encoded = classifier.predict(text)
        y_dict = convert_predictions_to_dict(y_raw.ravel())

    json_response = render_template('response.json',
                                    status="SUCCESS",
                                    review_raw=text,
                                    review_preprocessed=text_preprocessed,
                                    truth=truth,
                                    review_encoded=json.dumps(text_encoded.ravel().tolist()),
                                    rating=y_unencoded,
                                    prediction_raw=json.dumps(y_dict),
                                    timestamp=datetime.now().strftime(TIMESTAMP))
    app.logger.debug(json_response)

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


# @api.route('/history/api/v1.0', methods=['GET'])
def get_history():
    results = PredictionHistory.query.all()

    app.logger.debug(type(results))
    resp = '[ ' + ", ".join(str(result) for result in results) + " ]"

    # return results[0].to_json()
    return resp


def convert_predictions_to_dict(d: list):
    return {str(i + 1): str(d[i]) for i in range(len(d))}


# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
