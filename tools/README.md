# Tools

This directory contains a number of tools to generated features and run various models

There are a number of utilities that I wrote but the 4 main program you need are here. You should execute these in order:
* tsv_to_csv.py - convert Amazon files to CSV
* amazon_review_preprocessor.py - pre-process review text
* feature_generator.py - generate features based on pre-processed text
* run_classifiers.py - train models 


# tsv_to_csv.py

Original Amazon files were in tsv format. Pandas was not able to parse this properly. I used the following tool to covert it into csv format so Pandas can read the file(s) property.

### Usage

```buildoutcfg
(capstone) bash-3.2$ python tsv_to_csv.py --help
usage: tsv_to_csv.py [-h] [-l LOGLEVEL] [-f FUNCTION] [-s SAMPLING_PARAM]
                     infile outfile

Conver TSV to CSV

positional arguments:
  infile                Input TSV file
  outfile               Output CSV file

optional arguments:
  -h, --help            show this help message and exit
  -l LOGLEVEL, --loglevel LOGLEVEL
                        log level
  -f FUNCTION, --function FUNCTION
                        functions available: tsv_to_csv (default: tsv_to_csv)
  -s SAMPLING_PARAM, --sampling_param SAMPLING_PARAM
                        sampling param. For simple sampler - this means the
                        frequency to keep. Max 100. default = 0
```

# amazon_review_preprocessor.py

Use this file to pre-process Amazon review files

* make everything lowercase
* remove newlines
* remove amazon tags - amazon embeds these [[VIDDEO:dsfljlsjf]] and [[ASIN:sdfjlsjdfl]] tags that need to be removed
* remove html tags - line breaks, etc are represented in reviews as HTML tags
* remove accent characters
* expand contractions - expands contractions like he's but needs to be done before special charaters because we want to expand don't into do not for our text processing
* remove special characters - anything that is not alphanumeric or spaces
* remove stop words - see text_util.py for stop words that I removed from nltk stop words because I think they will be important for sentiment analysis
* lemmatize words - lemmatize words using wordnet




### Usage

```buildoutcfg
(capstone) bash-3.2$ python amazon_review_preprocessor.py --help
usage: amazon_review_preprocessor.py [-h] [-o OUTDIR] [-l LOGLEVEL] [-r] [-c]
                                     datafile

positional arguments:
  datafile              source data file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -l LOGLEVEL, --loglevel LOGLEVEL
                        log level
  -r, --retain          specifieds which columns to keep - NOT YET IMPLEMENTED
  -c, --convert         convert to csv
```

# feature_generator.py

Use this to generate various feature files that can be used in our models 

Supported:
* BoW
* TF-IDF
* Word2Vec
* FastText

### Usage
```buildoutcfg
(capstone) bash-3.2$ python feature_generator.py --help
usage: feature_generator.py [-h] [-l LOGLEVEL] [-r REPORTDIR] [-o OUTDIR]
                            config_file

Takes pre-processed files and generate feature files

positional arguments:
  config_file           file with parameters to drive the permutations

optional arguments:
  -h, --help            show this help message and exit
  -l LOGLEVEL, --loglevel LOGLEVEL
                        log level ie, DEBUG
  -r REPORTDIR, --reportdir REPORTDIR
                        report directory, default ../reports
  -o OUTDIR, --outdir OUTDIR
                        output director
```


# run_classifiers.py

This is a config based program that will run various models for you. I wrote this because my jupyter notebook kept crashing initially

Output is recored into a report file that can be analyzed in Jupyter notebook

Supported Classification Models:
* LR - Logistic Regression
* LRB - Logistic Regression (sample_weight='balanced')
* DT - Decision Tree
* DTB - Decision Tree (sample_weights='balanced')
* RF - Random Forest
* KNN - KNearestNeighbor 
* GB - Gradient Boosting
* lGBM - lightGBM
* XGB - XGBoost

### Usage


```buildoutcfg
(capstone) bash-3.2$ python run_classifiers.py --help
usage: run_classifiers.py [-h] [-l LOGLEVEL] [-r REPORTDIR] [--noreport]
                          [--lr_iter LR_ITER] [--n_jobs N_JOBS]
                          [--neighbors NEIGHBORS] [--radius RADIUS]
                          [--lr_c LR_C] [--epochs EPOCHS]
                          config_file

Run classifiers no feature files

positional arguments:
  config_file           file with parameters to drive the permutations

optional arguments:
  -h, --help            show this help message and exit
  -l LOGLEVEL, --loglevel LOGLEVEL
                        log level ie, DEBUG
  -r REPORTDIR, --reportdir REPORTDIR
                        report directory, default ../reports
  --noreport            do not generate report
  --lr_iter LR_ITER     number of iterations for LR. default 100
  --n_jobs N_JOBS       number of cores to use. default -1
  --neighbors NEIGHBORS
                        number of neighbors for KNN
  --radius RADIUS       radius for radius neighbor classification. default 30
  --lr_c LR_C           c parameter for LR. default 1.0
  --epochs EPOCHS       epoch for deep learning. Default 1
```


