#!/bin/bash
#
# Since I am running deep learning notebooks in google colab,
# I found manually copying and pasting file to google drive so they can be available in Google Colab
# a bit cumbersome. Created this script so I use Google Drive Sync to do this for me instead
#

usage() {
    echo "`basename $0`: [-n debug] [-u no_util]"
    echo "examples:"
    echo "  $0 -n -u"
}

NO_UTIL=""
while getopts "du" opt; do
    case $opt in
      d ) echo "Entering DEBUG mode" && DEBUG_FLAG="-n";;
      u ) NO_UTIL="true";;
      : ) echo "Option -"$OPTARG" requires an argument." >&2
          usage
          exit 1;;
      * ) echo "Invalid option: -"$OPTARG"" >&2
          usage
          exit 1;;
    esac
done

# sync all util file to google colab
echo ""
if [ "x${NO_UTIL}" == "xtrue" ]; then
    echo "Skip syncing util to google drive"
else
    echo "Syncing util to google drive..."
    rsync -rauv ${DEBUG_FLAG} --delete ../util/*.py ~/Google\ Drive/Springboard/capstone/util/
fi

# sync report files and notebooks from colab to git repot
echo ""
echo "Pulling down reports..."
rsync -rauv ${DEBUG_FLAG} --exclude=*test* --exclude=bak* ~/Google\ Drive/Springboard/capstone/reports/*.csv ../reports/

echo ""
echo "Pulling down network history files..."
rsync -rauv ${DEBUG_FLAG} --exclude=*test* --exclude=bak* ~/Google\ Drive/Springboard/capstone/reports/*history.pkl ../reports/

# sync report files and notebooks from colab to git repot
#echo "pulling down history files..."
#rsync -rauv ${DEBUG_FLAG} ~/Google\ Drive/Springboard/capstone/models/*history.pkl ../models/

# actually this is not a good idea - google colab notebooks are slightly different format
#echo "syncing colab notebooks to git..."
#rsync -rauv --exclude 5.0-deep-learning-summary.ipynb ~/Google\ Drive/Colab\ Notebooks/amazon-review/*.ipynb ../notebooks/deep_learning
#echo "syncing deep learning summary notebook to git..."
#rsync -rauv -n ~/Google\ Drive/Colab\ Notebooks/amazon-review/5.0-deep-learning-summary.ipynb ../notebooks/


