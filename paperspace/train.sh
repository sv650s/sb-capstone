#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing
# Free instance: https://docs.paperspace.com/gradient/instances/free-instances?_ga=2.254671808.999355169.1587737794-211442023.1587536380
#       C3 (CPU) or P4000 (GPU)
#    --container vtluk/paperspace-tf-gpu:1.0 \

usage() {
    echo "`basename $0`: [-a learning_rate] [-b batch_size] [-c lstm_cells] [-d dropout_rate] [-e epochs] [-l log_level]"
    echo "               [-m train_embeddings] [-p patience] [-r recurrent_dropout_rate] [-t machine_type] <network_type> <sample size>"
    echo "Parameter(s):"
    echo "  sample_size:                size of data set to train. available values: test, 50k, 100k, 200k, 500k, 1m, 2m, 4m, all"
    echo "Options:"
    echo "  -a learning_rate size:      learning_rate. Default 0.001"
    echo "  -b batch size:              batch size for training. Default 32"
    echo "  -c lstm_cells:              number of LSTM cells used for training. Default 128"
    echo "  -d dropout_rate:            dropout rate for LSTM network. Default 0"
    echo "  -e epochs:                  max number of epochs for training. Default 20"
    echo "  -l log_level:               log level for logging. Default INFO"
    echo "  -n enable_bidirectional:    Enable bidirectional network. Default False"
    echo "  -m train_embeddings:        Sets embedding layer to trainable. Default False"
    echo "  -p patience:                patience for early stopping. Default 4"
    echo "  -r recurrent_dropout_rate:  recurrent dropout rate for LSTM cells. Default 0"
    echo "  -t machine_type:            Gradient machine type. Options C3 (CPU) or P4000 (GPU). Default P4000"
    echo "Example:"
    echo "  ./train.sh LSTM 1m"
    echo "  ./train.sh -e 40 -d 0.2 GRU 1m"
}

if [ $# -lt 1 ]; then
    echo -e "ERROR: missing required parameter: <network_type> <sample_size>"
    usage
    exit 1
fi


# set variables
lstm_cells=128
lstm_cells_opt="-c ${lstm_cells} "
machine_type="P4000"
batch_size=32
batch_size_opt="-b ${batch_size} "
bidirectional_opt=
bidirectional_name=
unbalance_class_weights_opt=
balance_name="B"
train_embeddings_opt=
dropout_rate=0.0 && dropout_rate_opt="-d ${dropout_rate} "
recurrent_dropout_rate=0.0 && recurrent_dropout_rate_opt="-r ${recurrent_dropout_rate} "
learning_rate=0.001 && learning_rate_opt="-a ${learning_rate} "
resume_model_file_opt=
version=1
version_opt="-v ${version} "

while getopts :a:b:c:d:e:l:np:r:s:t:uv: o
   do
     case $o in
        a) learning_rate="$OPTARG" ; learning_rate_opt="-a ${learning_rate} " ;;
        b) batch_size="$OPTARG" && batch_size_opt="-b ${batch_size} " ;;
        c) lstm_cells="$OPTARG" && lstm_cells_opt="-c ${lstm_cells} ";;
        d) dropout_rate="$OPTARG" && dropout_rate_opt="-d ${dropout_rate} ";;
        e) epochs="$OPTARG" ;;
        l) log_level="$OPTARG" ;;
        n) bidirectional_opt="-n" && bidirectional_name="bi";;
        m) train_embeddings_opt="-m" ;;
        p) patience="$OPTARG" ;;
        r) recurrent_dropout_rate="$OPTARG" && recurrent_dropout_rate_opt="-r ${recurrent_dropout_rate} " ;;
        s) resume_model_file="$OPTARG" && resume_model_file_opt="-s ${resume_model_file} " ;;
        t) machine_type="$OPTARG" ;;
        u) unbalance_class_weights_opt="-u" && balance_name="" ;;
        v) version="$OPTARG " && version_opt="-v ${version}" ;;
        *) usage && exit 0 ;;                     # display usage and exit
     esac
   done

shift $((OPTIND-1))
network_type=$1
sample_size=$2
echo "Network type: ${network_type}"
echo "Sample size: ${sample_size}"

#if [ "x${lstm_cells}" == "x" ]; then
#    lstm_cells_opt=""
#else
#    lstm_cells_opt="-c ${lstm_cells} "
#fi
#if [ "x${dropout_rate}" == "x" ]; then
#    dropout_rate_opt=""
#else
#    dropout_rate_opt="-d ${dropout_rate} "
#fi
if [ "x${epochs}" == "x" ]; then
    epochs_opt=""
else
    epochs_opt="-e ${epochs} "
fi
if [ "x${log_level}" == "x" ]; then
    log_level_opt=""
else
    log_level_opt="-l ${log_level} "
fi
if [ "x${machine_type}" == "x" ]; then
    machine_type_opt=""
else
    machine_type_opt="-m ${machine_type} "
fi
if [ "x${patience}" == "x" ]; then
    patience_opt=""
else
    patience_opt="-p ${patience} "
fi
#if [ "x${recurrent_dropout_rate}" == "x" ]; then
#    recurrent_dropout_rate_opt=""
#else
#    recurrent_dropout_rate_opt="-r ${recurrent_dropout_rate} "
#fi


tf_version=`grep FROM.*tensorflow Dockerfile  | awk -F: '{print $2}'`
UTIL_ORIG="../util"
UTIL_DEST="train/util"
echo "Syncing util..."
rsync -rauv --delete --exclude="__pycache__" ${UTIL_ORIG}/*.py ${UTIL_DEST}/

# test-LSTMB16-1x16-dr0-rdr0-batch0-lr01-glove_with_stop_nonlemmatized-sampling_none-test-review_body

if [ ${sample_size} == "test" ]; then
    model_basename="test-${bidirectional_name}LSTM${balance_name}${lstm_cells}-"
else
    model_basename="${bidirectional_name}LSTM${balance_name}${lstm_cells}-"
fi
model_basename="${model_basename}1x${lstm_cells}-"
model_basename="${model_basename}dr`echo ${dropout_rate} | awk -F\. '{print $2}'`-"
model_basename="${model_basename}rdr`echo ${recurrent_dropout_rate} | awk -F\. '{print $2}'`-"
model_basename="${model_basename}batch${batch_size}-lr`echo ${learning_rate} | awk -F\. '{print $2}'`-"
# TODO: parameterize feature_set_name
model_basename="${model_basename}glove_with_stop_nonlemmatized-"
model_basename="${model_basename}sampling_none-"
model_basename="${model_basename}${sample_size}-"
model_basename="${model_basename}review_body-"
model_basename="${model_basename}v${version}"

echo "Running python with following command"
echo "python train/train.py -i /storage -o /artifacts ${batch_size_opt}${bidirectional_opt}${lstm_cells_opt}${dropout_rate_opt}${epochs_opt}${log_level_opt}${patience_opt}${recurrent_dropout_rate_opt}${unbalance_class_weights_opt}${train_embeddings_opt}${learning_rate_opt}${resume_model_file_opt}${version_opt} -t ${network_type} ${sample_size}"
echo "basename: ${model_basename}"
echo "modelPath: /artifacts/models/${model_basename}"
echo "tf_version: ${tf_version}"

gradient experiments run singlenode \
    --name ${model_basename} \
    --projectId pr1cl53bg \
    --machineType ${machine_type} \
    --container vtluk/paperspace-experiment:${tf_version} \
    --command "python train/train.py -i /storage -o /artifacts ${batch_size_opt}${bidirectional_opt}${lstm_cells_opt}${dropout_rate_opt}${epochs_opt}${log_level_opt}${patience_opt}${recurrent_dropout_rate_opt}${unbalance_class_weights_opt}${train_embeddings_opt}${learning_rate_opt}${resume_model_file_opt}${version_opt} -t ${network_type} ${sample_size}" \
    --workspace . \
    --modelType Tensorflow \
    --modelPath "/artifacts/models/${model_basename}"
#    --modelPath "/artifacts/models/"


echo "python train/train.py -i /storage -o /artifacts ${batch_size_opt}${bidirectional_opt}${lstm_cells_opt}${dropout_rate_opt}${epochs_opt}${log_level_opt}${patience_opt}${recurrent_dropout_rate_opt}${unbalance_class_weights_opt}${train_embeddings_opt}${learning_rate_opt}${resume_model_file_opt}${version_opt} -t ${network_type} ${sample_size}"
echo "basename: ${model_basename}"
echo "modelPath: /artifacts/models/${model_basename}"


