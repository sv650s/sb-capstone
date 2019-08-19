#!/bin/bash
#
# https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
#
# Use this script to set up and tear down GCP resources using gsutil
#
# NOTES:
# You can stream logs from the command line by running:
#  $ gcloud app logs tail -s default
#
# To view your application in the web browser run:
#  $ gcloud app browse

source set_vars.sh

usage() {
    echo "Usage: $0 [-disu] [-t <image_id>] [version]"
    echo "     -d - deploy container. This should be runned after you have pushed a container in -t"
    echo "     -i - initialize project"
    echo "     -s - shutdown project"
    echo "     -t - tag docker image and upload <image_id>"
    echo "     -u - update container for deployment"
}


version="v1"
while getopts dist:u o
do    case "$o" in
    d)    deploy="x";;
    i)    init="x";;
    s)    shutdown="x";;
    t)    tag="x" && image_id="$OPTARG";;
    u)    update="x";;
    [?])    usage && exit 1;;
    esac
done
shift $((OPTIND-1))

if [ $# -gt 0 ]; then
    version=$1
fi

echo "version: $version"

tag() {
    # push docker container
    # https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
    # docker build -t gcr.io/${PROJECT_ID}/hello-app:v1 .
    docker tag $image_id ${IMAGE_REPO}:$version

    gcloud auth configure-docker --quiet

    echo "pushing docker image: ${IMAGE_REPO}:$version"
    docker push ${IMAGE_REPO}:$version
}


init() {
    # update gcloud tools
    gcloud components update

    # set current project in gsutil
    gcloud config set project ${PROJECT_ID}
    gcloud config set compute/zone ${ZONE}

    gsutil ls gs://${BUCKET_NAME}
    if [ $? -eq 1 ]; then
        # create bucket
        echo "${BUCKET_NAME} not found. Creating file bucket"
        gsutil mb gs://${BUCKET_NAME}/
    fi


}





shutdown() {

    echo "deleting service: ${DEPLOYMENT_NAME}"
    kubectl delete service ${DEPLOYMENT_NAME}

    echo "deleting cluster name: ${CLUSTER_NAME}"
    gcloud container clusters delete ${CLUSTER_NAME}

    # tear down the GCP bucket
    echo "deleting storage bucket: ${BUCKET_NAME}"
    gsutil rm -r gs://$BUCKET_NAME

}


deploy() {
    echo "creating cluster: ${CLUSTER_NAME}"
    gcloud beta container clusters create ${CLUSTER_NAME} --num-nodes=1 --enable-stackdriver-kubernetes
    if [ $? -eq 1 ]; then
        echo "ERROR: failed to create cluster"
        exit 1
    fi

    gcloud compute instances list
    if [ $? -eq 1 ]; then
        echo "ERROR: failed to get instances list"
        exit 1
    fi

    echo "creating deployment: ${DEPLOYMENT_NAME}"
    kubectl create deployment ${DEPLOYMENT_NAME} --image=${IMAGE_REPO}:$version
    if [ $? -eq 1 ]; then
        echo "ERROR: failed to create deployment"
        exit 1
    fi

    echo "getting pods"
    kubectl get pods


    echo "exposing ${DEPLOYMENT_NAME}"
    kubectl expose deployment ${DEPLOYMENT_NAME} --type=LoadBalancer --port 80 --target-port 5000
    if [ $? -eq 1 ]; then
        echo "ERROR: failed to expose deployment"
        exit 1
    fi

    kubectl get service
}

# update a deployment with new image
update() {
    echo "Run and update the following command"
    echo "kubectl set image deployment/${DEPLOYMENT_NAME} ${SERVICE}=${IMAGE_REPO}:$version"
    kubectl set image deployment/${DEPLOYMENT_NAME} ${SERVICE}=${IMAGE_REPO}:$version
}

create_cluster() {

    gcloud container clusters create capstone-cluster --num-nodes=1

}

if [ "x$init" == "xx" ]; then
    init
elif [ "x$shutdown" == "xx" ]; then
    shutdown
elif [ "x$tag" == "xx" ]; then
    tag
elif [ "x$deploy" == "xx" ]; then
    deploy
elif [ "x$update" == "xx" ]; then
    update
fi


