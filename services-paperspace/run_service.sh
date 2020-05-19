#!/bin/bash

/etc/init.d/mysql start

echo "sleeping for 10s"
sleep 10

debug="true"

export FLASK_APP=reviews.py
export FLASK_ENV=development

echo "Starting flask server"
flask run --host 0.0.0.0

