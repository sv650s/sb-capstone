# Notesbook

This directory contains notebooks to evaluate various models.

I explored traditional ML models as well as deep learning models for this project.


Unfortunately when I first started on this project jupyter notebook was not stable and my model runs would hang and never finish. So I created python programs to do the bulk of running the models.

Deep learning notebooks were executed on Google Colab and checked into the repository since they require a GPU

Configuration files and reports are checked into the repository so we can reproduce the results later.

For details on how to run model, see documentation in the *tools* directory

Notebooks numbered in the order they were executed


## Directory Structure

```buildoutcfg
├── exploratory - all exploratory notebooks
├── deep_learning - google colab notebooks used for deep learning
├── tests - scratch pad space for tests (you can ignore this)
```


## Running from command line

Sometime I re-run notebooks via the command line. You can do that using *runAllNotebooks.sh*


### Usage
```bash
$ ./runNotebooks.sh -h
Use this script to run notebooks from the command line. By default, notebooks will be overwritten with the output (including errors)
If -n is note specified, this will run all notebooks in the directory
./runNotebooks.sh: [-d run all notebooks in debug] [-n <notebook> specific notebook or a pattern for notebooks] [-c when running in debug mode, us this to delete temp notebooks]
Example:
  ./runNotebooks.sh
  ./runNotebooks.sh -n notebook.ipynb
  ./runNotebooks.sh -n 3*.ipynb
```
