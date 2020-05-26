# Amazon Review Classification Service

This is implemented as a flask server using flask, SQLAlchemy, and flask-restplus (Swagger UI) 

This Project is deployed as a Docker image onto Paperspace machine so that it's accessible to the outside.

## Files and Directories

* util/ - python utilities. These are copied over from ../util
* templates/ - flask templates (ie, json api responses)
* config.py - configuration settings for Flask app
* review.py - main Flask application
* run_service.sh - startup script for our service
* requirements.txt - pip requirements file. Used to install python packages in Docker container
* sync_util.sh - rsync necessary python utilities from ../util

# Prerequisites


# Project Setup

# Creating Paperspace Machine

In Gradient Paperspace console, set up the following VM:

* machine type: C2
* public IP address: enabled
* machine template: Ubuntu 18.04
* location: NY2 - save location as our GPU experiments so the machine has access to models created in /storage directory

# Paperspace VM Machine One Time Setup

## Ubuntu Setup

Once the machine is provisioned and running. Log into the machine terminal and run the following command:

```bash
# install utilities
sudo apt-get install tree

# install python3.7 - ubuntu comes with python3.6 as a default
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

# make python 3.7 default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

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

## MySql Setup

In order for the flask application to work properly, you must create the database in MySql before. This is a one-time setup per container

The first time you run your docker container. Run the following commands

From Paperspace VM Machine, start interactive shell into Docker container

```bash
> docker exec -it <container id> /bin/bash
```

Connect to MySql database

```bash
# sudo to root
sudo su -
# connect to mysql database
mysql
```

Create reviews user and capstonedb database

```sql
GRANT ALL PRIVILEGES ON *.* TO 'reviews'@'%' IDENTIFIED BY 'password';
CREATE DATABASE capstonedb;
FLUSH PRIVILEGES;
```

Restart Docker container (in a separate shell)
```bash
docker container stop <container id>
docker container start -ia <container id>
```

## Running Service


### Run service locally

```bash
./sync_util.sh
docker-compose build
docker-compose up
```

#### Local service URL

```log
http://localhost:5000
```

### Paperspace

#### Copying Models On Paperspace

Flask application the following file/directory structure to load models:
```bash
├── <model name 1>-v<model version 1>
│   ├── <model name 1>-v<model version 1>-model.json
│   ├── <model name 1>-v<model version 1>-weights.h5
│   └── <model name 1>-v<model version 1>-tokenizer.pkl
├── <model name 2>-v<model version 2>
│   ├── <model name 2>-v<model version 2>-model.json
│   ├── <model name 2>-v<model version 2>-weights.h5
│   └── <model name 2>-v<model version 2>-tokenizer.pkl
```

The following directory structure/files should be copied to the ~/models directory on the paperspace VM machine

```bash
rsync -rauvh --progress . paperspace@<machine public ip>:~/models/
```

##### For Local Development

Create a 'models' directory in this folder and create the same directory structure as above. When running local version of docker container, a volume mount will be created to map the folder into Docker container

### Run service on Paperspace

Building and running Flask server 
```bash
# copy docker-compose file to machine
scp docker-compose-paperspace.yml paperspace@<machine public ip>:~/docker-compose.yml
# ssh to machine
ssh paperspace@<machine public ip>
docker-compose build
docker-compose up
```

#### Paperspace URL

```log
http://<machine public ip>:5000
```


# References
[Paperspace Machine User Guide](https://docs.paperspace.com/gradient/machines/using-machines)
[Paperspace Startup Script Guide](https://github.com/Paperspace/paperspace-node/blob/master/scripts.md)
[MySql setup on ubuntu](https://support.rackspace.com/how-to/install-mysql-server-on-the-ubuntu-operating-system/)


# TODO's

* database one-time manual setup steps to Dockerfile
* mysql is not starting up using init.d properly. We get around this by starting it in our startup script
* move database password out of config.py
* split out reviews.py into various python files
* update docker image to use TF 2.0.1 (same as google Colab)

# Common Docker Commands

Build New Docker Image
```bash
docker-compose build
```

Create new container based on built image
```bash
docker-compose up
```

Start docker container with attach logs
```bash
docker container start -ia <container id>
```

Stop docker container
```bash
docker container stop <container id>
```

Interactive shell into docker container
```bash
docker exec -it <container id> /bin/bash
```

Look at docker logs
```bash
docker logs -a <container id>
```

Clear out non-running docker containers
```bash
docker container prune -f
```

Clear out docker images without containers
```bash
docker image prune -a
```


# MySql Commands

Connect to database
```bash
mysql -h <db ip> -u <username> -p <password>
```

Show available databases
```sql
show databases;
```

Use database
```sql
use <database name>;
```
