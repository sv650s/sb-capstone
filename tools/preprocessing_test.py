# using this file to test pre-processing
# I noticed the following when manually reviewing results
#       time -> ti ame
#       sometimes -> someti ame
#       compliments -> compli aments
#
import sys
sys.path.append('../')

from util.TextPreprocessor import TextPreprocessor

import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)



STOP_WORDS_TO_REMOVE=[
    'no',
    'not',
    'do',
    'don',
    "don't",
    'does',
    'did',
    'does',
    'doesn',
    "doesn't",
    'should',
    'very',
    'will'
    ]

if __name__ == '__main__':
    df = pd.DataFrame(['time', 'sometimes', 'compliments', "I'm a person"]).rename({0: "text"}, axis=1)


    tp = TextPreprocessor(text_columns=["text"],
                          # columns_to_drop=['marketplace', 'vine', 'verified_purchase', 'customer_id', 'review_id',
                          #                  'product_id', 'product_parent', 'product_title', 'product_category'],
                          stop_word_remove_list=STOP_WORDS_TO_REMOVE,
                          retain_original_columns=True,
                          remove_stop_words=False,
                          lemmatize_text=False)
    df = tp.preprocess_data(df)

    print(df.head())