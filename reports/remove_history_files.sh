#!/bin/bash
#
# remove history files locally from https://colab.research.google.com/drive/1DdYPOcXxg7CmRFtpjHdmFUtPj7t2GGSq#scrollTo=LNwamrGU8Y_l


files='LSTMB128-1x128-glove_with_stop_nonlemmatized-sampling_none-49887-100-review_body-history.pkl LSTMB128-1x128-glove_with_stop_nonlemmatized-sampling_none-99772-100-review_body-history.pkl LSTMB128-1x128-glove_with_stop_nonlemmatized-sampling_none-199538-100-review_body-history.pkl LSTMB128-1x128-dr0-rdr2-batch32-lr001-glove_with_stop_nonlemmatized-sampling_none-500k-review_body-v1-history.pkl LSTMB128-1x128-glove_with_stop_nonlemmatized-sampling_none-997682-100-review_body-history.pkl LSTMB128-1x128-dr0-rdr2-batch32-lr01-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl LSTMB128-1x128-dr0-rdr2-batch32-lr01-glove_with_stop_nonlemmatized-sampling_none-997682-100-review_body-history.pkl LSTMB16-1x16-dr0-rdr2-batch32-lr001-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl LSTMB16-1x16-dr2-rdr2-batch128-lr001-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl LSTMB128-1x128-dr0-rdr0-batch32-lr01-glove_with_stop_nonlemmatized-sampling_none-500k-review_body-v1-history.pkl LSTMB128-1x128-dr0-rdr0-batch32-lr01-glove_with_stop_nonlemmatized-sampling_none-1m-review_body-v1-history.pkl LSTMB128-1x128-dr0-rdr0-batch32-lr001-glove_with_stop_nonlemmatized-sampling_none-500k-review_body-v1-history.pkl LSTMB128-1x128-dr0-rdr0-batch32-lr001-glove_with_stop_nonlemmatized-sampling_none-1m-review_body-v1-history.pkl LSTMB128-1x128-dr0-rdr0-batch128-lr01-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl LSTMB128-1x128-dr0-rdr0-batch128-lr01-glove_with_stop_nonlemmatized-sampling_none-1m-review_body-v1-history.pkl LSTMB128-1x128-dr2-rdr2-batch128-lr01-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl LSTMB128-1x128-dr2-rdr2-batch128-lr01-glove_with_stop_nonlemmatized-sampling_none-1m-review_body-v1-history.pkl LSTMB128-1x128-dr2-rdr2-batch128-lr01-glove_with_stop_nonlemmatized-sampling_none-all-review_body-v1-history.pkl LSTMB128-1x128-dr2-rdr2-batch128-lr001-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl LSTMB128-1x128-dr2-rdr2-batch32-lr001-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl LSTMB128-1x128-dr0-rdr0-batch128-lr001-glove_with_stop_nonlemmatized-sampling_none-498831-100-review_body-history.pkl'

for i in $files; do
    echo $i
    if [ -f $i ]; then
        git rm $i
        rm $i
    fi

done
