#!/bin/bash
# run unit tests and coverage report


#python -m unittest tests/TestTextUtil.py
#python -m pytest tests/*

pytest --cov=. --cov-config=.coveragerc --cov-report  html --capture=no tests/*
