#!/bin/bash
# run unit tests and coverage report


#python -m unittest tests/test_text_util.py
#python -m pytest tests/*

pytest --cov=util --cov-config=.coveragerc --cov-report  html tests/*
