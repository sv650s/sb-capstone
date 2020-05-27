"""
predict.py
===============================
Utility program that uses the model_builder to load a model, feed in a bunch of review bodies
and expected outputs and do model inference and print resutls on review bodies

usage: python predict.py
"""
import sys
sys.path.append('../../')

import logging

import util.model_builder as mb

logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
    " [%(lineno)d] - %(message)s"
logging.basicConfig(level = logging.WARNING, format = LOG_FORMAT)

reviews = {
    "This is the most amazing product. must buy": 5,
    "Great case. Great customer service": 5,
    "Works great": 5,
    "I really love it. Only thing I don't like is the color": 4,
    "This is just ok. Wish it worked better": 3,
    "I like it but wish it worked better": 3,
    "didn't work well but I liked the color": 2,
    "This didn't work at all. Do not buy": 1
}

models = [
    "GRU16-1x16-random_embedding-sampling_none-199538-100-review_body",
    "LSTMB128-1x128-dr2-rdr2-batch128-lr01-glove_with_stop_nonlemmatized-sampling_none-1m-review_body-tf2.2.0"
]

if __name__ == "__main__":

    # if len(sys.argv) < 3:
    #     print(f"ERROR: usage: python {sys.argv[0]} <truth> <review body>")
    #     exit(1)
    # label = int(sys.argv[1])
    # review_body = sys.argv[2]

    for model in models:
        mb.ModelFactory.init_factory("../models", 100, "json")

        for review_body, label in reviews.items():

            logger.info(f"label: {label} review body: [{review_body}]")




            classifier = mb.ModelFactory.get_instance().get_model(model, 1)

            y_unencoded, y_raw, text_preprocessed, text_sequence = classifier.predict(review_body)
            print(f'Prediction for models: {model}')
            print(f'\treview preprocessed: {text_preprocessed}')
            print(f'\treview sequence: {text_sequence}')
            print(f'\treversed sequence: {classifier.preprocessor.tokenizer.sequences_to_texts(text_sequence)}')
            print(f'\tprediction raw: {y_raw}')
            print(f'\tprediction unencoded: {y_unencoded}')
            print(f'\texpected: {label}')
