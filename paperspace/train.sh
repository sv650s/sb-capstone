#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing
# Free instance: https://docs.paperspace.com/gradient/instances/free-instances?_ga=2.254671808.999355169.1587737794-211442023.1587536380
#       C3 (CPU) or P4000 (GPU)
#    --container vtluk/paperspace-tf-gpu:1.0 \

usage() {
    echo "`basename $0`: [-b batch_size] [-c lstm_cells] [-d dropout_rate] [-e epochs] [-l log_level] [-m machine_type]"
    echo "               [-p patience] [-r recurrent_dropout_rate] <sample size>"
    echo "Parameter(s):"
    echo "  sample_size:                size of data set to train. available values: test, 50k, 100k, 200k, 500k, 1m, 2m, 4m, all"
    echo "Options:"
    echo "  -b batch size:              batch size for training. Default 32"
    echo "  -c lstm_cells:              number of LSTM cells used for training. Default 128"
    echo "  -d dropout_rate:            dropout rate for LSTM network. Default 0"
    echo "  -e epochs:                  max number of epochs for training. Default 20"
    echo "  -l log_level:               log level for logging. Default INFO"
    echo "  -m machine_type:            Gradient machine type. Options C3 (CPU) or P4000 (GPU). Default P4000"
    echo "  -n enable_bidirectional:    Enable bidirectional network. Default False"
    echo "  -p patience:                patience for early stopping. Default 4"
    echo "  -r recurrent_dropout_rate:  recurrent dropout rate for LSTM cells. Default 0"
    echo "Example:"
    echo "  ./train.sh 1m"
    echo "  ./train.sh -e 40 -d 0.2 1m"
}

if [ $# -lt 1 ]; then
    echo -e "ERROR: missing required parameter: sample_size"
    usage
    exit 1
fi


machine_type="P4000"
bidirectional_opt=
bidirectional_name=
unbalance_class_weights_opt=
balance_name="B"
while getopts :b:c:d:e:l:m:np:r:u o
   do
     case $o in
        b) batch_size="$OPTARG" ;;
        c) lstm_cells="$OPTARG" ;;
        d) dropout_rate="$OPTARG" ;;
        e) epochs="$OPTARG" ;;
        l) log_level="$OPTARG" ;;
        m) machine_type="$OPTARG" ;;
        n) bidirectional_opt="-n"; bidirectional_name="bi";;
        p) patience="$OPTARG" ;;
        r) recurrent_dropout_rate="$OPTARG" ;;
        u) unbalance_class_weights_opt="-u"; balance_name="" ;;
        *) usage && exit 0 ;;                     # display usage and exit
     esac
   done

shift $((OPTIND-1))
sample_size=$1
echo "Sample size: ${sample_size}"

if [ "x${batch_size}" == "x" ]; then
    batch_size_opt=""
else
    batch_size_opt="-b ${batch_size}"
fi
if [ "x${lstm_cells}" == "x" ]; then
    lstm_cells_opt=""
else
    lstm_cells_opt="-c ${lstm_cells}"
fi
if [ "x${dropout_rate}" == "x" ]; then
    dropout_rate_opt=""
else
    dropout_rate_opt="-d ${dropout_rate}"
fi
if [ "x${epochs}" == "x" ]; then
    epochs_opt=""
else
    epochs_opt="-e ${epochs}"
fi
if [ "x${log_level}" == "x" ]; then
    log_level_opt=""
else
    log_level_opt="-l ${log_level}"
fi
if [ "x${machine_type}" == "x" ]; then
    machine_type_opt=""
else
    machine_type_opt="-m ${machine_type}"
fi
if [ "x${patience}" == "x" ]; then
    patience_opt=""
else
    patience_opt="-p ${patience}"
fi
if [ "x${recurrent_dropout_rate}" == "x" ]; then
    recurrent_dropout_rate_opt=""
else
    recurrent_dropout_rate_opt="-r ${recurrent_dropout_rate}"
fi


echo "Running python with following command"
echo "python train/train.py -i /storage -o /artifacts ${batch_size_opt} ${bidirectional_opt} ${lstm_cells_opt} ${dropout_rate_opt} ${epochs_opt} ${log_level_opt} ${patience_opt} ${recurrent_dropout_rate_opt} ${sample_size}" \

gradient experiments run singlenode \
    --name ${bidirectional_name}LSTM${balance_name}-${sample_size} \
    --projectId pr1cl53bg \
    --machineType ${machine_type} \
    --container vtluk/paperspace-tf-gpu:1.0 \
    --command "python train/train.py -i /storage -o /artifacts ${batch_size_opt} ${bidirectional_opt} ${lstm_cells_opt} ${dropout_rate_opt} ${epochs_opt} ${log_level_opt} ${patience_opt} ${recurrent_dropout_rate_opt} ${unbalance_class_weights_opt} ${sample_size}" \
    --workspace .


