#!/bin/bash

if [[ `hostname | awk -F. '{print $2}'` == "local" ]]; then
    source gcp_vars.sh
fi

debug="false"

export FLASK_APP=reviews.py
export FLASK_ENV=development

flask run --host 0.0.0.0

