#!/bin/bash
#
# goes through all pre-processed files in our dataset and generate word-count files for them all
#

#list="../dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-test-preprocessed.csv"

(
    cd ../tools
    for i in ../dataset/amazon_reviews/*preprocessed.csv; do
    # for i in $list; do
        python generate_word_count.py -d lemmatized $i
        # echo $i;
    done
)
