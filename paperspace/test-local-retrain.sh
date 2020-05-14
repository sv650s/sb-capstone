#!/bin/bash
# test that train.py works by locally running keras model
# this version will actually load an existing model to train
#

MODEL_FILE="models/test-LSTMB16-1x16-dr0-rdr0-batch128-lr01-glove_with_stop_nonlemmatized-sampling_none-test-review_body-v1-model.h5"

rm -rf /tmp/models
rm -rf /tmp/reports
./syncUtil.sh && python train/train.py -i ../dataset -o /tmp -b 128 -c 16 -a 0.01 -e 3 -l DEBUG -s ${MODEL_FILE} test
