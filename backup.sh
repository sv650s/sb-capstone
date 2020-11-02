#!/bin/sh

# TODO: we want to sync with delete except for probably the dataset directory
# in which case we probably don't want to delete
rsync -rauv --exclude amazon_reviews --exclude dataset * /Volumes/vince/0_springboard/capstone/
# only back up here if I have it mounted
if [ -d /Volumes/vince\ 1 ]; then
    rsync -rauv --exclude amazon_reviews * /Volumes/vince\ 1/0_springboard/capstone/
fi
