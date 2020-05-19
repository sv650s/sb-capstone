# Amazon Review Classification Service

This is implemented as a flask server using flask, SQLAlchemy, and flask-restplus (Swagger UI) 

This Project is deployed as a Docker image onto Paperspace machine so that it's accessible to the outside.

## Files and Directories

* util/ - python utility classes
* templates/ - flask templates
* tests/ - unit tests
* config/ - contains a sample json config for one of our models (for development)
* gcp.sh - utility script to make GCP project setup easier
* gcp_vars.sh - environment variables for GCP project - will want to edit this with info from your project
* review.py - main Flask application
* run_service.sh - script to run our flask server. This will also be used by our Docker container
* requirements.txt - pip requirements file

# Prerequisites


# Project Setup

```bash
python -m venv .
source bin/activate
pip install -r requirements.txt

```

Optionally, you can use install virtualenv and create a python virtual environment before installing requirements for the project


## Paperspace Machine Setup

Paperspace machine setup:
* machine type: C2
* public IP address: enabled
* machine template: Ubuntu 18.04
* location: NY2 - save location as our GPU experiments so the machine has access to models created in /storage directory

### Ubuntu Setup

Run the following command after initial machine has been provisioned

```bash
# install utilities
sudo apt-get install tree

# install python3
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
apt-get dist-upgrade
sudo apt install python3.7

# remove python 2
sudo apt autoremove

# install pip for python3
sudo apt-get install python3-pip
python3.7 -m pip install pip

# make python 3 default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# install virtual env
sudo apt-get install python3.7-venv

# install Docker
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
# verify that docker is properly installed and running
sudo docker run hello-world
```


# Running Service


### Run flask service via Docker

This method does the following:
* loads models from GCP bucket using GCPModelBuilder
* volume mounts the current directory to docker container so any changes done in local file system will be picked up on the flask server


```bash
source gcp_vars.sh
docker-compose build
docker-compose up

```


# Running and deploying service on GCP

Do the following to deploy your service as well as any other associated files and containers to GCP

```bash
source gcp_vars.sh
docker-compose build # if you haven't built the latest version of your container already
./gcp.sh start_all 

```

To get your docker image hash to deploy run:
```bash
docker images
```

# Accessing Your Service

I used restplus which gives you a Swagger UI. You can use this to test your service


Hit the following URL's on your browser

## Local and Local Docker Image
```bash
http://localhost:5000
```

## GCP
```bash
http://<your k8 cluster external IP>
```
This runs on port 80 so you don't have to specify port. You can get get the external IP by running:


# References
[Paperspace Machine User Guide](https://docs.paperspace.com/gradient/machines/using-machines)
[Paperspace Startup Script Guide](https://github.com/Paperspace/paperspace-node/blob/master/scripts.md)
