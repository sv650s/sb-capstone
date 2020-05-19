#!/bin/bash

# TODO: uncomment this when mysql is implemented
/etc/init.d/mysql start
echo "sleeping for 5s"
#sleep 5

debug="true"

export FLASK_APP=reviews.py
export FLASK_ENV=development

echo "Starting flask server"
flask run --host 0.0.0.0

