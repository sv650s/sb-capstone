#!/bin/bash

rsync -rauv --exclude=__pycache__ . mini:~/Dropbox/0_springboard/capstone-old/
