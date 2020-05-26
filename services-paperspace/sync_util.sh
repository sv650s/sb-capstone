#!/bin/bash
#
# sets up the util directory and create hard links to originals in ../util directory
#

FROM_DIR=../util
TO_DIR=util

files="__init__.py model_builder.py python_util.py amazon_util.py text_util.py tf2_util.py service_preprocessor.py contraction_map.py"

if [ ! -d ${TO_DIR} ]; then
    mkdir ${TO_DIR}
fi

(
    cd ${TO_DIR}

    for file in ${files}; do
#        rsync -auq ../${FROM_DIR}/${file} .
        rsync -auv ../${FROM_DIR}/${file} .
#        echo "Linking ${file}"
#        ln ../${FROM_DIR}/${file}
    done

)