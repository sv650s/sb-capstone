#!/bin/bash
#
# create pre-processed files with the following parameters
#   don't remove stop words
#   don't lemmatize text
#   retain original columns so we can debug later
#

usage() {
    echo "`$0`: [-d debug]"
    echo "      generates pre-processed files for all datasets - don't remove stop words, don't lemmatize text, retain orignal columns"
}



DEBUG=true


if [ "x$DEBUG" == "xtrue" ]; then
    files="dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-test.csv"
else
    files="dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-50k.csv dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-100k.csv dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-200k.csv dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-500k.csv dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-1m.csv dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-2m.csv
dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-4m.csv dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-all.csv"
fi


for file in ${files}; do
    echo "pre-process file: ${file}"
    python amazon_review_preprocessor.py -r -p with_stop_nonlemmatized -s -e ../${file}
    echo "done pre-processing file: ${file}"
done

