#!/bin/bash

debug="false"

while getopts d o
do	case "$o" in
	d)	debug="true";;
	[?])	print >&2 "Usage: $0 [-d]"
		exit 1;;
	esac
done
shift $OPTIND-1


export FLASK_APP=reviews.py

if [ $debug == "true" ]; then
    export FLASK_ENV=development
fi

flask run

