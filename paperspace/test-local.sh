#!/bin/bash
# test that train.py works by locally running keras model
#


rm -rf /tmp/models
rm -rf /tmp/reports
./syncUtil.sh && python train/train.py -i ../dataset -o /tmp -b 128 -c 16 -a 0.01 -e 3 -l DEBUG  test
