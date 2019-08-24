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
    echo "     create_cluster - deploy container. This should be runned after you have pushed a container in -t"
    echo "     delete_db - deletes database"
    echo "     init - initialize project"
    echo "     shutdown_cluster - shutdown project"
    echo "     tag - tag docker image and upload. Requires additional <version> <image_id>"
    echo "     update_image - update container for deployment"
}


version="v1"
#while getopts cdist:u o
#do    case "$o" in
#    c)    copy="x";;
#    d)    create_cluster="x";;
#    i)    init="x";;
#    s)    shutdown_cluster="x";;
#    t)    tag="x" && image_id="$OPTARG";;
#    u)    update_image="x";;
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
    if [ "x$version" == "x" -o "x$image_id" == "x" ]; then
        echo -e "ERROR: tag requireds <version> and <image_id> parameters"
        usage
        exit 1
    fi

    docker tag $image_id ${IMAGE_REPO}:$version

    gcloud auth configure-docker --quiet

    echo -e "\npushing docker image: ${IMAGE_REPO}:$version"
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
        echo -e "${BUCKET_NAME} not found. Creating file bucket"
        gsutil mb gs://${BUCKET_NAME}/
        echo -e "set bucket read permission to public read"
        gsutil defacl set public-read gs://${BUCKET_NAME}
    fi


}

copy() {
    # copy model files to GCP
    gsutil cp models/* gs://${BUCKET_NAME}
    gsutil cp config/* gs://${BUCKET_NAME}
}

delete_bucket() {

    # tear down the GCP bucket
    echo -e "deleting storage bucket: ${BUCKET_NAME}"
    gsutil rm -r gs://$BUCKET_NAME

}


shutdown_cluster() {

    echo -e "deleting service: ${DEPLOYMENT_NAME}"
    kubectl delete service ${DEPLOYMENT_NAME}

    echo -e "deleting cluster name: ${CLUSTER_NAME}"
    gcloud container clusters delete ${CLUSTER_NAME}

}


create_cluster() {
    if [ x$version == "x" ]; then
        "ERROR: missing version"
        usage
        exit 1
    fi

    echo -e "creating cluster: ${CLUSTER_NAME}"
    gcloud beta container clusters create ${CLUSTER_NAME} --num-nodes=1 --enable-stackdriver-kubernetes
    if [ $? -eq 1 ]; then
        echo -e "ERROR: failed to create cluster"
        exit 1
    fi

    gcloud compute instances list
    if [ $? -eq 1 ]; then
        echo -e "ERROR: failed to get instances list"
        exit 1
    fi
    sleep 3

    echo -e "\ncreating deployment: ${DEPLOYMENT_NAME}"
    kubectl create deployment ${DEPLOYMENT_NAME} --image=${IMAGE_REPO}:$version
    if [ $? -eq 1 ]; then
        echo -e "ERROR: failed to create deployment"
        exit 1
    fi

    echo -e "\ngetting pods"
    kubectl get pods


    echo -e "\nexposing ${DEPLOYMENT_NAME}"
    kubectl expose deployment ${DEPLOYMENT_NAME} --type=LoadBalancer --port 80 --target-port 5000
    if [ $? -eq 1 ]; then
        echo -e "ERROR: failed to expose deployment"
        exit 1
    fi


    # we need to assign a static ip to our compute instance since we are accessing
    # MySql via a public IP address and we need to allow traffic to come in from that IP
    instance_name=`gcloud compute instances list | tail -1 | awk '{print $1}'`
    echo -e "\nsetting static ip for compute instance ${instance_name}..."
    gcloud compute addresses create ${instance_name} --region ${REGION}
    if [ $? -eq 1 ]; then
        echo -e "ERROR: requesting static ip for compute instance"
        exit 1
    fi

    # get external IP of our instance
    instance_external_ip=`gcloud compute instances list | tail -1 | awk '{print $5}'`
    my_ip=`dig +short myip.opendns.com @resolver1.opendns.com`
    echo -e "\nadding external IP $external_ip to database"
    gcloud sql instances patch ${DB_INSTANCE_NAME} --authorized-networks=${instance_external_ip},${my_ip}


    kubectl get service



}

# update a deployment with new image
update_image() {
    echo -e "Run and update the following command"
    echo -e "kubectl set image deployment/${DEPLOYMENT_NAME} ${SERVICE}=${IMAGE_REPO}:$version"
    kubectl set image deployment/${DEPLOYMENT_NAME} ${SERVICE}=${IMAGE_REPO}:$version
}

#create_cluster() {
#
#    gcloud container clusters create ${CLUSTER_NAME} --num-nodes=1
#
#}

create_database() {
    # reference: https://cloud.google.com/sql/docs/mysqlo/create-instance

    # get my IP address
    my_ip=`dig +short myip.opendns.com @resolver1.opendns.com`

    # create mysql database
    # https://cloud.google.com/sql/docs/mysql/create-instance
    echo -e "creating database..."
    gcloud sql instances create ${DB_INSTANCE_NAME} --tier=${DB_MACHINE_TYPE} --region=${REGION} --authorized-networks=${my_ip}
    if [ $? -gt 1 ]; then
        echo -e "error creating database"
    fi
    # configure public IP for insance
    # https://cloud.google.com/sql/docs/mysql/configure-ip
    echo -e "setting username password..."
    gcloud sql users set-password root % --instance=${DB_INSTANCE_NAME} --password 'freel00k'
    if [ $? -gt 1 ]; then
        echo -e "error setting password"
    fi

    echo -e "\nconfiguring public ip access..."
    gcloud sql instances patch ${DB_INSTANCE_NAME} --assign-ip
    if [ $? -gt 1 ]; then
        echo -e "error assigning IP"
    fi
    gcloud sql instances describe ${DB_INSTANCE_NAME}
#    gcloud sql instances patch ${db_instance_name} --authorized-networks=${my_ip}

    # configure instance to use SSL
    echo -e "\nrequire ssl connections to database..."
    gcloud sql instances patch ${DB_INSTANCE_NAME} --require-ssl
    if [ $? -gt 1 ]; then
        echo -e "error configuring ssl"
    fi


    # TODO: in order to connect to MySQL data base we need to assign a static ip to our compute instance
    # set compute instance to have static IP
    # add compute instance static IP to accepted DB connection


#   mysql -uroot -p -h 35.247.24.229 \
#    --ssl-ca=server-ca.pem --ssl-cert=client-cert.pem \
#    --ssl-key=client-key.pem


}

delete_database() {
    # deletes database from GCP
    # https://cloud.google.com/sql/docs/mysql/delete-instance
    echo -e "deleting database..."
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
elif [ "x${command}" == "xshutdown_cluster" ]; then
    shutdown_cluster
elif [ "x${command}" == "xtag" ]; then
    tag
elif [ "x${command}" == "xcreate_cluster" ]; then
    create_cluster
elif [ "x${command}" == "xupdate_image" ]; then
    update_image
else
    echo -e "command $command not found"
    usage
    exit 1
fi


