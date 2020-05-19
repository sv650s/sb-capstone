#!/bin/bash
#
# connect to mysql DB on gcp
#

source gcp_vars.sh

mysql -h $DB_IP -u root --password --ssl-cert=credentials/client-cert.pem --ssl-key=credentials/client-key.pem
