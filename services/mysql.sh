#!/bin/bash

source gcp_var.sh

mysql -h $DB_IP -u root --password --ssl-cert=credentials/client-cert.pem --ssl-key=credentials/client-key.pem
