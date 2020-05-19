#!/bin/bash


debug="false"

export FLASK_APP=reviews.py
export FLASK_ENV=development

flask run --host 0.0.0.0

