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

# MySql Set

In order for the flask application to work properly, you must create the database in MySql before.

The first time you run your docker container. Run the following commands

## From Paperspace Machine

Start interactive shell into Docker container

```bash
> docker exec -it <container id> /bin/bash
```

## Inside Docker Container

Connect to MySql database

```bash
# sudo to root
sudo su -
# connect to mysql database
mysql
```

## MySql shell

Create capstonedb database

```sql
GRANT ALL PRIVILEGES ON *.* TO 'reviews'@'%' IDENTIFIED BY 'password';
CREATE DATABASE capstonedb;
FLUSH PRIVILEGES;
```

## docker-compose

scp docker-compose file from this directory to Paperspace VM Machine

```bash
scp docker-compose-paperspace.yml paperspace@<machine ip>:/home/paperspace/
```

# Running Service


## Run service locally

```bash
./sync_util.sh
docker-compose build
docker-compose up
```

### Local service URL

http://localhost:5000


## Run service on Paperspace

```bash
ssh paperspace@<machine public ip>
docker-compose build
docker-compose up
```

### Local service URL

http://<machine public ip>:5000


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


# References
[Paperspace Machine User Guide](https://docs.paperspace.com/gradient/machines/using-machines)
[Paperspace Startup Script Guide](https://github.com/Paperspace/paperspace-node/blob/master/scripts.md)
[MySql setup on ubuntu](https://support.rackspace.com/how-to/install-mysql-server-on-the-ubuntu-operating-system/)


# Notes

start mysql /etc/init.d/mysql start

/usr/sbin/mysqld --basedir=/usr --datadir=/var/lib/mysql --plugin-dir=/usr/lib/mysql/plugin --log-error=/var/log/mysql/error.log --pid-file=/var/run/mysqld/mysqld.pid --socket=/var/run/mysqld/mysqld.sock --port=3306 --log-syslog=1 --log-syslog-facility=daemon --log-syslog-tag=


# MySql Commands

Connect to database
```bash
mysql -h <db ip> -u <username> -p <password>
```

Show databases
```sql
show databases;
```
