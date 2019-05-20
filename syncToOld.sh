#!/bin/bash

rsync -rauv --exclude=__pycache__ . 192.168.1.88:~/Dropbox/0_springboard/capstone-old/
