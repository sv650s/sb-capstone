#!/bin/bash

# Use this script to set up and tear down GCP resources using gsutil
#
# NOTES:
# You can stream logs from the command line by running:
#  $ gcloud app logs tail -s default
#
# To view your application in the web browser run:
#  $ gcloud app browse


export PROJECT="sb-capstone"
export PROJECT_ID="786682764664"
export ZONE="us-central1-a"
export BUCKET_NAME="${PROJECT}-bucket"


while getopts st o
do	case "$o" in
	s)	setup="x";;
	t)	teardown="x";;
	[?])	print >&2 "Usage: $0 [-s] [-t]"
		exit 1;;
	esac
done
#shift $OPTIND-1



setup() {
    # update gcloud tools
    gcloud components update

    # set current project in gsutil
    gcloud config set project ${PROJECT_ID}

    # install python 3 extensions
    # https://cloud.google.com/appengine/docs/standard/python3/quickstart
    gcloud components install app-engine-python


    gsutil ls gs://${BUCKET_NAME}
    if [ $? -eq 1 ]; then
        # create bucket
        echo "${BUCKET_NAME} not found. Creating file bucket"
        gsutil mb gs://${BUCKET_NAME}/
    fi

}



teardown() {
    # tear down the GCP bucket
    gsutil rm -r gs://$BUCKET_NAME
}


deploy() {
    # deploy app using app.yaml
    gcloud app deploy
}

if [ "x$setup" == "xx" ]; then
    setup
elif [ "x$teardown" == "xx" ]; then
    teardown
fi


