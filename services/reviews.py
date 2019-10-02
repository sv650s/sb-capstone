#!/anaconda3/envs/capstone/bin/flask

from __future__ import absolute_import, division, print_function, unicode_literals

from flask import Flask
from flask import abort
from flask import render_template, Blueprint
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import validates
from sqlalchemy.types import Text
from flask_restplus import Api, Resource, fields

from datetime import datetime
import json
import tensorflow as tf
from config import Config
from util.model_util import ModelFactory
from logging.config import dictConfig
import traceback2 as traceback
from flask_restplus import abort
from flask.logging import default_handler
import logging

TIMESTAMP = "%Y-%m-%d %H:%M:%S"

# TODO: error handling
# https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-vii-error-handling


# configure logging
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
api = Api(app, version=app.config['VERSION'], title="Vince's Amazon Review Classifier",
          description='Predict product rating based on user review input')
db = SQLAlchemy(app)

logging.basicConfig(level=logging.DEBUG)

for logger in (
    logging.getLogger('util.AmazonTextNormalizer'),
    logging.getLogger('util.gcp_file_util'),
    logging.getLogger('util.model_util'),
    logging.getLogger('util.preprocessor'),
    logging.getLogger('util.python_util'),
    logging.getLogger('util.text_util'),
    logging.getLogger('util.tf2_util')
):
    logger.addHandler(default_handler)


class Prediction(db.Model):

    # set to 63535 later - this is equivalent to TEXT type in mysql, the rest willl be TINYTEXT type
    MAX_REVIEW_LENGTH = 63535
    id = db.Column(db.Integer, primary_key=True)
    input_raw = db.Column(Text, nullable=False)
    input_preprocessed = db.Column(Text, nullable=True)
    input_encoded = db.Column(Text, nullable=True)
    class_expected = db.Column(db.Integer, nullable=False)
    class_predicted = db.Column(db.Integer, nullable=False)
    class_predicted_raw = db.Column(db.String(256), nullable=False)  # json of softmax output
    model_version = db.Column(db.String(256), nullable=False)
    model_name = db.Column(db.String(256), nullable=False)
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    status = db.Column(db.String(256), nullable=False)

    def __repr__(self):
        return f'<ID: {self.id}\tEXPECTED: {self.class_expected}\tPREDICTED: {self.class_predicted}'

    def to_json(self):
        """
        returns json representation of prediction history
        :return:
        """
        return json.loads(str(self))

    def __str__(self):
        """
        Returns a json string representation of the history object
        :return:
        """
        return render_template('response.json',
                               status=self.status,
                               review_raw=self.input_raw,
                               review_preprocessed=self.input_preprocessed,
                               truth=self.class_expected,
                               review_encoded=self.input_encoded,
                               rating=self.class_predicted,
                               prediction_raw=self.class_predicted_raw,
                               model=self.model_name,
                               version=self.model_version,
                               timestamp=self.created.strftime(TIMESTAMP))

    @validates('class_predicted_raw', 'model_version', 'model_name', 'status')
    def validate_string_fields(self, key, value):
        """
        Truncates string fields if it's longer that what we specified
        :param key:
        :param value:
        :return:
        """
        max_len = getattr(self.__class__, key).prop.columns[0].type.length
        app.logger.debug(f'max_len {max_len}')
        app.logger.debug(f'len(value) {len(value)}')
        if value and len(value) > max_len:
            return value[:max_len]
        return value


app.logger.info("creating database...")
db.create_all()
db.session.commit()
app.logger.info("finished creating database...")


def get_factory():
    return ModelFactory


def get_model_cache_json():
    model_names = get_factory().model_list()
    json_reponse = render_template('response-cache.json',
                                   status="SUCCESS",
                                   timestamp=datetime.now().strftime(TIMESTAMP),
                                   size=len(model_names),
                                   model_names=json.dumps(model_names))
    return json.loads(json_reponse)


@api.route('/model/cache', endpoint='model')
class ModelCache(Resource):

    @api.response(200, 'Success')
    @api.doc(description="Clear our model cache")
    def delete(self):
        return json.loads(clear_model_cache()), 200

    @api.response(200, 'Success')
    @api.doc(description="List all models in the cache")
    def get(self):
        return get_model_cache_json()


@api.route('/predict/<string:model>/<string:version>')
@api.doc(params={'model': 'model name - ie, GRU',
                 'version': 'model version - ie, v1.0'})
class ModelPrediction(Resource):
    # expected input
    predict_model = api.model("predict", {
        "review": fields.String(title="Review", required=True),
        "product_rating": fields.Integer(title="class rating - 1 to 5", required=True)
    })

    @api.expect(predict_model, validate=True)
    @api.response(201, 'Success')
    @api.response(400, 'Error while predicting results')
    @api.response(404, 'Validation error or cannot find model')
    @api.doc(description="Run the provided review through our model to see it's prediction")
    # @api.marshal_with(model)
    def post(self, model, version):
        input = api.payload
        app.logger.debug(f'input {input}')
        if input['product_rating'] < 1 or input['product_rating'] > 5:
            abort(400, 'product_rating must be between 1 and 5')

        return predict_reviews(model, version, input['review'], input['product_rating'])


@api.route('/predict/history')
class ModelHistory(Resource):

    @api.response(200, 'Success')
    @api.doc(description='Retrieves a history of predictions')
    def get(self):
        return get_history_json()


def clear_model_cache():
    """
    Reset and clear all cached models
    :return:
    """
    get_factory().clear()
    return get_model_cache_json()


def predict_reviews(model_name, version, text, truth):
    """
    Call our model based on the name and version and review text
    :param model_name:
    :param version:
    :param text:
    :param truth:
    :return:
    """
    # if not request.args or not 'review' in request.args or not 'truth' in request.args:
    #     abort(400)
    app.logger.info(f'Tensorflow version: {tf.__version__}')

    # TODO: un-hard code this - get this from the URL
    classifier = get_factory().get_model(model_name, version)
    # json_response = None
    prediction = None
    if classifier:
        try:
            y_unencoded, y_raw, text_preprocessed, text_sequence = classifier.predict(text)
            y_dict = convert_predictions_to_dict(y_raw.ravel())

            status = "SUCCESS"
            # save response to DB
            prediction = Prediction(input_raw=text,
                                    input_preprocessed=text_preprocessed,
                                    input_encoded=json.dumps(text_sequence),
                                    class_expected=truth,
                                    class_predicted=y_unencoded,
                                    class_predicted_raw=json.dumps(y_dict),
                                    model_name=model_name,
                                    model_version=version,  # get this from the URL
                                    status=status)
            db.session.add(prediction)
            db.session.commit()

            json_response = prediction.to_json()

        except:
            error_message = traceback.format_exc()
            app.logger.error(error_message)
            json_response = json.loads(render_template('response.json',
                                                       status="FAILED",
                                                       error_message="Model not found",
                                                       review_raw=text,
                                                       truth=truth,
                                                       model=model_name,
                                                       version=version,
                                                       timestamp=datetime.now().strftime(TIMESTAMP)))
            abort(400, error_message, custom=json_response)
    else:
        template_str = render_template('response.json',
                                       status="FAILED",
                                       error_message="Model not found",
                                       review_raw=text,
                                       truth=truth,
                                       model=model_name,
                                       version=version,
                                       timestamp=datetime.now().strftime(TIMESTAMP))
        app.logger.debug(f'template_str {template_str}')
        json_response = json.loads(template_str)
        app.logger.debug(f'response {json_response}')
        abort(404, 'model not found', custom=json_response)

    app.logger.debug(json_response)

    return json_response, 201


def get_history_json():
    """
    returns json representation of history object
    format will be:
    [
        prediction1 json,
        prediction2 json,
        .
        .
    ]
    :return:
    """
    results = Prediction.query.order_by(Prediction.created.desc()).all()

    app.logger.debug(type(results))
    resp = '[ ' + ", ".join(str(result) for result in results) + " ]"

    # return results[0].to_json()
    return json.loads(resp)


def convert_predictions_to_dict(d: list):
    """
    Converts the raw list of floats from softmax function into a dictionary
    values for dictionary have to be converted to string in order for it to be convertible to json
    :param d:
    :return:
    """
    return {str(i + 1): str(d[i]) for i in range(len(d))}


# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
