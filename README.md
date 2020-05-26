# Amazon Review Classification

For this project, we will use NLP to see if we can correctly predict star ratings for Amazon reviews based on the review body

# Environment Setup

You will need to install anaconda python package manager on your machine. Once you have done that you can execute the following command to set up the environment

```bash
conda env create --file environments.yml
```


# Directory Structure


```buildoutcfg
├── config - config files for python programs
├── docs - sphinx documentation
├── dataset - sample dataset files. This is empty
├── images - images/diagrams for notebooks and README's
├── notebooks - jupyter notebooks
├── reports - reports from python programs
├── services - final Flask API 
├── tests - pytests and coverage report
├── tools - python programs
├── util - utility classes and modules
└── runUnitTests.sh - run unit tests and create coverage report
```


NOTE: Because of github size limit, the data files are not checked into the repository

You can down the entire dataset directly here: 

https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt