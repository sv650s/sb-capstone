#!/bin/bash
# build and push docker image to docker hub

# get version from Dockerfile based on tensorflow version
tf_version=`grep FROM.*tensorflow Dockerfile  | awk -F: '{print $2}' | awk -F- '{print $1}'`

docker image build --tag vtluk/paperspace-tf-gpu:${tf_version} .
docker image push vtluk/paperspace-tf-gpu:${tf_version}
