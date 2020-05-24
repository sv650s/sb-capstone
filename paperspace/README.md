# paperspace-amazon-review
Repo for paperspace DL model training


# Create Docker Image for Training

Base image is tensorflow/tensorflow with GPU (2.0)
Additional python packages are installed via pip using docker-requirements.txt file

## Build and push docker image for training

./docker.sh

## Running Docker Image locally (interactive mode)

docker run -it --name tf vtluk/paperspace-tf:1.0 /bin/bash


# Train Model

Use the following script to traing models on gradient paperspace.

P4000 GPU instance will be created for model training

```bash
$ ./train.sh -?
train.sh: [-a learning_rate] [-b batch_size] [-c lstm_cells] [-d dropout_rate] [-e epochs] [-l log_level]
               [-m train_embeddings] [-p patience] [-r recurrent_dropout_rate] [-t machine_type] <sample size>
Parameter(s):
  sample_size:                size of data set to train. available values: test, 50k, 100k, 200k, 500k, 1m, 2m, 4m, all
Options:
  -a learning_rate size:      learning_rate. Default 0.001
  -b batch size:              batch size for training. Default 32
  -c lstm_cells:              number of LSTM cells used for training. Default 128
  -d dropout_rate:            dropout rate for LSTM network. Default 0
  -e epochs:                  max number of epochs for training. Default 20
  -l log_level:               log level for logging. Default INFO
  -n enable_bidirectional:    Enable bidirectional network. Default False
  -m train_embeddings:        Sets embedding layer to trainable. Default False
  -p patience:                patience for early stopping. Default 4
  -r recurrent_dropout_rate:  recurrent dropout rate for LSTM cells. Default 0
  -t machine_type:            Gradient machine type. Options C3 (CPU) or P4000 (GPU). Default P4000
Example:
  ./train.sh 1m
  ./train.sh -e 40 -d 0.2 1m
```

## Training Locally

```bash
$ python train/train.py --help

usage: train.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR] [-d DROPOUT_RATE]
                [-r RECURRENT_DROPOUT_RATE] [-f FEATURE_COLUMN]
                [-t TRUTH_LABEL_COLUMN] [-n] [-p PATIENCE] [-c LSTM_CELLS]
                [-e EPOCHS] [-b BATCH_SIZE] [-l LOGLEVEL]
                sample_size

positional arguments:
  sample_size           Sample size (ie, 50k, test)

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        input directory. Default /storage/data
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory. Default /artifacts
  -d DROPOUT_RATE, --dropout_rate DROPOUT_RATE
                        dropout rate. Default 0
  -r RECURRENT_DROPOUT_RATE, --recurrent_dropout_rate RECURRENT_DROPOUT_RATE
                        recurrent dropout rate. NOTE: will not be able to
                        cuDNN if this is set. Default 0
  -f FEATURE_COLUMN, --feature_column FEATURE_COLUMN
                        feature column. Default review_body
  -t TRUTH_LABEL_COLUMN, --truth_label_column TRUTH_LABEL_COLUMN
                        label column. Default star_rating
  -n, --bidirectional   label column. Default star_rating
  -p PATIENCE, --patience PATIENCE
                        patience. Default = 4
  -c LSTM_CELLS, --lstm_cells LSTM_CELLS
                        Number of LSTM cells. Default = 128
  -e EPOCHS, --epochs EPOCHS
                        Max number epochs. Default = 20
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Training batch size. Default = 32
  -l LOGLEVEL, --loglevel LOGLEVEL
                        log level
```


### Test if everything is working locally

./syncUtil.sh &&  python train/train.py -i ../dataset -o /tmp -b 128 -c 16 -e 1 -l DEBUG test

## Artifacts

Model training will generate the following artifacts on paperspace

```log
├── reports - csv report, missing words csv, network history
├── models - model binary, model json, model weights
```



  
