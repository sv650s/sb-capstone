#!/bin/bash
# build and push docker image to docker hub

tf_version=`grep ^FROM Dockerfile | awk -F: '{print $2}'`

docker image build --tag vtluk/paperspace-experiment:${tf_version} .
docker image push vtluk/paperspace-experiment:${tf_version}
