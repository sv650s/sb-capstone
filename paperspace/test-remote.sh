#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing
# Free instance: https://docs.paperspace.com/gradient/instances/free-instances?_ga=2.254671808.999355169.1587737794-211442023.1587536380
#       C3 (CPU) or P4000 (GPU)
#    --container vtluk/paperspace-tf-gpu:1.0 \


UTIL_ORIG="../util"
UTIL_DEST="train/util"
echo "Syncing util..."
rsync -rauv --delete --exclude="__pycache__" ${UTIL_ORIG}/*.py ${UTIL_DEST}/
sleep 2

./train.sh -e 3 -c 16 -a 0.01 -b 128 -d 0.0 -r 0.0 -t GRU test


