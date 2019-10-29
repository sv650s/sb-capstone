#!/bin/sh
# Extracts the header and particular line numbers from a file and dumps it into a new file
#

usage() {
    echo "$0: <start line number> <end line number> <infile> <outfile>"
}

if [ $# -lt 4 ]; then
    usage
    exit 1
fi

start=$1
end=$2
infile=$3
outfile=$4


head -1 $infile > $outfile
sed -n "$start,${end}p" $infile >> $outfile


