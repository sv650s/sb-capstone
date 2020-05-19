#!/bin/bash
#
# sync files that we need for our services from ../util to the util directory in this module
#


files="__init__.py model_builder.py python_util.py AmazonTextNormalizer.py text_util.py tf2_util.py gcp_file_util.py"

if [ ! -d util ]; then
    mkdir util
fi

for file in ${files}; do
    rsync -auv ../util/${file} util/
done