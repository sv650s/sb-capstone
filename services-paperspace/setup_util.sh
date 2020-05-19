#!/bin/bash
#
# sets up the util directory and create hard links to originals in ../util directory
#

FROM_DIR=../util
TO_DIR=util

files="__init__.py model_builder.py python_util.py AmazonTextNormalizer.py text_util.py"

if [ ! -d ${TO_DIR} ]; then
    mkdir ${TO_DIR}
fi

(
    cd ${TO_DIR}

    for file in ${files}; do
        ln ${FROM_DIR}/${file}
    done

)