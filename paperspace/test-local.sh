#!/bin/bash
# test that train.py works by locally running keras model
#


rm -rf /tmp/models
rm -rf /tmp/reports
./syncUtil.sh && python train/train.py -i ../dataset -o /tmp -b 128 -c 128 -d 0.0 -r 0.0 -a 0.01 -e 6 -p 4 -l DEBUG  test
