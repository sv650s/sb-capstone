#!/bin/bash
#
# use nbconvert to run all notebooks via the command line
# if you use -d, this will use the test dataset that should run a lot faster. You can use this to verify that
# all notebooks have the right syntax
#
# running in debug mode will not overwrite existing notebooks since we don't pass the --inplace paramter in
#

usage() {
  echo "Use this script to run notebooks from the command line. By default, notebooks will be overwritten with the output (including errors)"
  echo "If -n is note specified, this will run all notebooks in the directory"
  echo "$0: [-d run all notebooks in debug] [-n <notebook> specific notebook or a pattern for notebooks] [-c when running in debug mode, us this to delete temp notebooks]"
  echo "Example:"
  echo "  $0"
  echo "  $0 -n notebook.ipynb"
  echo "  $0 -n 3*.ipynb"
}

DEBUG="false"
CLEAN="false"
ALL="true"
notebooks=""
DEBUG_MSG=""

while getopts "chdn:" arg; do
  case $arg in
    h) usage && exit 0 ;;
    n) ALL="false" && notebooks=`ls $OPTARG` ;;
    d) DEBUG="true" ;;
    c) CLEAN="true" ;;
    ?) usage && exit 1 ;;
  esac
done

if [ $ALL == "false" -a "x$notebooks" == "x" ]; then
  echo "Notebook parameter required"
  usage
  exit 1
elif [ $ALL == "true" ]; then
  notebooks=`ls *.ipynb | grep -v nbconvert`
fi

log_file="$0.log"

# delete existing log file
if [ -f $log_file ]; then
  rm $log_file
fi

if [ $DEBUG == "true" ]; then
  export IPYNB_DEBUG="True"
  echo "Running in DEBUG mode" | tee -a $log_file
  DEBUG_MSG="in DEBUG mode"
  rm ../reports/summary-test.csv
fi


# notebooks starting with 4 are our regular ML classification notebooks
for notebook in $notebooks; do
    echo "" | tee -a $log_file
    echo "`date` Running $notebook ${DEBUG_MSG}" | tee -a $log_file
    if [ $DEBUG == "true" ]; then
      # don't run inplace
      jupyter nbconvert --to notebook --allow-errors --ExecutePreprocessor.timeout=3600 --execute $notebook 2>&1 | tee -a $log_file
    else
      jupyter nbconvert --to notebook --allow-errors --inplace --ExecutePreprocessor.timeout=3600 --execute $notebook 2>&1 | tee -a $log_file
    fi
    echo "`date` Finished running $notebook" | tee -a $log_file
    echo "" | tee -a $log_file

    if [ $CLEAN == "true" ]; then
      echo "cleaning up temp notebook"
      rm `echo $notebook | awk -F\. '{print $1}'`.nbconvert.ipynb
    fi
done

unset IPYNB_DEBUG

