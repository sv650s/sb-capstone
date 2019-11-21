# Pyspark Environments

To train models with more data I used Pyspark since my sklearn programs would error out and just quit.

The following docker services are available:

* pyspark-juptyer - this is a pyspark environment with jupyter notebook environment installed


## Instructions

Before running the container, make sure you have pre-processed the raw amazon data files by running tsv_to_csv.py and amazon_review_preprocessor.py on the file that you need

```bash
docker-compose build
docker-compose up
```