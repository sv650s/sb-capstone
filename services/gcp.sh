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

source gcp_vars.sh

usage() {
    echo "Usage: $0 <cmd> [cmd params]"
    echo "Available cmd's"
    echo "     copy - copy model files to file bucket ${BUCKET_NAME}"
    echo "     create_db - creates database"
    echo "     deploy - deploy container. This should be runned after you have pushed a container in -t"
    echo "     delete_db - deletes database"
    echo "     init - initialize project"
    echo "     shutdown - shutdown project"
    echo "     tag - tag docker image and upload. Requires additional <version> <image_id>"
    echo "     update - update container for deployment"
}


version="v1"
#while getopts cdist:u o
#do    case "$o" in
#    c)    copy="x";;
#    d)    deploy="x";;
#    i)    init="x";;
#    s)    shutdown="x";;
#    t)    tag="x" && image_id="$OPTARG";;
#    u)    update="x";;
#    [?])    usage && exit 1;;
#    esac
#done
#shift $((OPTIND-1))

if [ $# -gt 0 ]; then
    command=$1
    if [ $# -gt 1 ]; then
        version=$2
    fi
    if [ $# -gt 2 ]; then
        image_id=$3
    fi
else
    echo "ERROR: missing commaand"
    usage
    exit 1
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
        echo "set bucket read permission to public read"
        gsutil defacl set public-read gs://${BUCKET_NAME}
    fi


}

copy() {
    # copy model files to GCP
    gsutil cp models/* gs://${BUCKET_NAME}
    gsutil cp config/* gs://${BUCKET_NAME}
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

create_database() {
    # reference: https://cloud.google.com/sql/docs/mysqlo/create-instance

    # get my IP address
    my_ip=`dig +short myip.opendns.com @resolver1.opendns.com`

    # create mysql database
    # https://cloud.google.com/sql/docs/mysql/create-instance
    echo "creating database..."
    gcloud sql instances create ${DB_INSTANCE_NAME} --tier=${DB_MACHINE_TYPE} --region=${REGION} --authorized-networks=${my_ip}
    if [ $? -gt 1 ]; then
        echo "error creating database"
    fi
    # configure public IP for insance
    # https://cloud.google.com/sql/docs/mysql/configure-ip
    echo "setting username password..."
    gcloud sql users set-password root % --instance=${DB_INSTANCE_NAME} --password 'freel00k'
    if [ $? -gt 1 ]; then
        echo "error setting password"
    fi

    echo "configuring public ip access..."
    gcloud sql instances patch ${DB_INSTANCE_NAME} --assign-ip
    if [ $? -gt 1 ]; then
        echo "error assigning IP"
    fi
    gcloud sql instances describe ${DB_INSTANCE_NAME}
#    gcloud sql instances patch ${db_instance_name} --authorized-networks=${my_ip}

    # configure instance to use SSL
    echo "configuring ssl..."
    gcloud sql instances patch ${DB_INSTANCE_NAME} --require-ssl
    if [ $? -gt 1 ]; then
        echo "error configuring ssl"
    fi


    # set compute instance to have static IP
    # add compute instance static IP to accepted DB connection


#   mysql -uroot -p -h 35.247.24.229 \
#    --ssl-ca=server-ca.pem --ssl-cert=client-cert.pem \
#    --ssl-key=client-key.pem


}

delete_database() {
    # deletes database from GCP
    # https://cloud.google.com/sql/docs/mysql/delete-instance
    echo "deleting database..."
    gcloud sql instances delete ${DB_INSTANCE_NAME}

}


if [ "x${command}" == "xcopy" ]; then
    copy
elif [ "x${command}" == "xcreate_db" ]; then
    create_database
elif [ "x${command}" == "xdelete_db" ]; then
    delete_database
elif [ "x${command}" == "xinit" ]; then
    init
elif [ "x${command}" == "xshutdown" ]; then
    shutdown
elif [ "x${command}" == "xtag" ]; then
    tag
elif [ "x${command}" == "xdeploy" ]; then
    deploy
elif [ "x${command}" == "xupdate" ]; then
    update
fi


