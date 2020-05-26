#!/bin/bash
# build and push docker image to docker hub

# get version from Dockerfile based on tensorflow version
tf_version=`grep FROM.*tensorflow Dockerfile  | awk -F: '{print $2}'`

docker image build --tag vtluk/paperspace-experiment:${tf_version} .
docker image push vtluk/paperspace-experiment:${tf_version}
