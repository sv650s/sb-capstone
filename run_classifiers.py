import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from models.ClassifierRunner import ClassifierRunner
from datetime import datetime
import pandas as pd
import logging
import argparse
import util.file_util as fu
import sys

# configure logger so we can see output from the classes
LOG_FORMAT='%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'

# set global variables

# I'm finding that running these models on my laptop takes forever and they are not finishing so I'm going to start
# with a really small file just to validate my code
#
# datafile was generated from amazon_review_preprocessing.ipynb - this file has 1k reviews randomly chosen
# from original file
# KEEP_COLUMNS = ["product_title", "helpful_votes", "review_headline", "review_body", "star_rating"]


# Configuration
# DATA_FILES = ["dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-tinyout.csv"]
# NEIGHBORS = [5] # default
# NEIGHBORS = [1, 3, 5, 7, 9, 11]

# Radius for RadiusNeighbor
# RADII = [5.0] # this is the lowest number I tried that was able to find a neighbor for review_headline
# RADII = [30.0] # this is the lowest number I tried that was able to find a neighbor for review_body
# RADII = [5.0, 7.0, 9.0, 11.0, 13.0]

# logistic regression settings
# C= [1.0] # default
# C = [0.2, 0.4, 0.6, 0.8, 1.0]

# N_JOBS=-1
# LR_ITER=500





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add a class column ')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    parser.add_argument("--noknn", help="don't do KNN", action='store_true')
    parser.add_argument("--nolr", help="don't do logistic regression", action='store_true')
    parser.add_argument("--norn", help="don't do radius neighbor", action='store_true')
    parser.add_argument("--noreport", help="don't do radius neighbor", action='store_true')
    parser.add_argument("--lr_iter", help="number of iterations for LR", default=300)
    parser.add_argument("--n_jobs", help="number of iterations for LR", default=-1)
    parser.add_argument("--neighbors", help="number of neighbors for KNN", default=5)
    parser.add_argument("--radius", help="radius for radius neighbor classification", default=30)
    parser.add_argument("--lr_c", help="c parameter for LR", default=1.0)
    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)


    log.debug(f'noknn={args.noknn}')
    log.debug(f'nolr={args.nolr}')
    log.debug(f'norn={args.norn}')
    log.debug(f'noreport={args.noreport}')

    # ready in config file
    # config file has the following format
    config_df = pd.read_csv(args.config_file)
    report_file_name = fu.get_report_filename(args.config_file)
    config_length = len(config_df)

    n_jobs = int(args.n_jobs)
    lr_iter = int(args.lr_iter)
    neighbors = int(args.neighbors)
    radius = int(args.radius)
    lr_c = int(args.lr_c)

    start_time = datetime.now()
    cr = ClassifierRunner(write_to_csv=not args.noreport, outfile=report_file_name)
    report_df = pd.DataFrame()
    for index, row in config_df.iterrows():

        data_dir = row["data_dir"]
        data_file = row["data_file"]
        class_column = row["class_column"]
        # description = row["description"]
        description = data_file.split(".")[0]
        log.debug(f'description {description}')

        infile = f'{data_dir}/{data_file}'
        log.info(f"loading file {infile}")
        X = pd.read_csv(infile)
        Y = X[class_column]
        X.drop(class_column, axis=1)

        log.info(f'shape of X: {X.shape}')
        log.info(f'shape of Y: {Y.shape}')


        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

        if not args.noknn:
                neigh = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=n_jobs)
                cr.addModel(neigh,
                            X_train,
                            Y_train,
                            X_test,
                            Y_test,
                            name="KNN",
                            description=description,
                            parameters={"n_jobs": n_jobs,
                                        "n_neighbors": neighbors})

        if not args.norn:
                rnc = RadiusNeighborsClassifier(radius=radius, n_jobs=n_jobs)
                cr.addModel(rnc,
                            X_train,
                            Y_train,
                            X_test,
                            Y_test,
                            name="RN",
                            description=description,
                            parameters={"n_jobs": n_jobs,
                                        "radius": radius} )

        if not args.nolr:
                lr = LogisticRegression(random_state=0, solver='lbfgs',
                                        multi_class='auto',
                                        max_iter=lr_iter, n_jobs=n_jobs, C=lr_c)
                cr.addModel(lr,
                            X_train,
                            Y_train,
                            X_test,
                            Y_test,
                            name="LR",
                            description=description,
                            parameters={"n_jobs": n_jobs,
                                        "c": lr_c,
                                        "max_iter": lr_iter} )

    report_df = cr.runAllModels()
    print(report_df.tail())


    log.info("Finished running all models")
    end_time = datetime.now()
    total_time = end_time - start_time
    log.info(f'Total time: {total_time}')
    print(report_df.head())

