#!/bin/bash
# test that train.py works by locally running keras model
#

epoch_opt=
while getopts :a:b:c:d:e:l:np:r:s:t:uv: o
   do
     case $o in
        a) learning_rate="$OPTARG" ; learning_rate_opt="-a ${learning_rate} " ;;
        b) batch_size="$OPTARG" && batch_size_opt="-b ${batch_size} " ;;
        c) lstm_cells="$OPTARG" && lstm_cells_opt="-c ${lstm_cells} ";;
        d) dropout_rate="$OPTARG" && dropout_rate_opt="-d ${dropout_rate} ";;
        e) epoch_opt="-e $OPTARG" ;;
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

rm -rf /tmp/models
rm -rf /tmp/reports
./syncUtil.sh && python train/train.py -i ../dataset -o /tmp -b 128 -c 128 -d 0.0 -r 0.0 -a 0.01 ${epoch_opt} -p 4 -l DEBUG  test
#./syncUtil.sh && python train/train.py -i ../dataset -o /tmp -b 128 -c 128 -d 0.0 -r 0.0 -a 0.01 -e 6 -p 4 -l DEBUG  test
