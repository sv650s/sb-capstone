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
import util.df_util as dfu

# configure logger so we can see output from the classes
LOG_FORMAT='%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)

def create_training_data(x:pd.DataFrame, class_column:str):
    """
    Take dataframe and:
    1. split between features and predictions
    2. create test and training sets
    :param x:
    :param class_column:
    :return:
    """

    y = x[class_column]
    x.drop(class_column, axis=1, inplace=True)

    log.info(f'shape of x: {x.shape}')
    log.info(f'shape of y: {y.shape}')

    return train_test_split(x, y, random_state=1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add a class column ')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    parser.add_argument("--noknn", help="don't do KNN", action='store_true')
    parser.add_argument("--nolr", help="don't do logistic regression", action='store_true')
    parser.add_argument("--norn", help="don't do radius neighbor", action='store_true')
    parser.add_argument("--noreport", help="don't do radius neighbor", action='store_true')
    parser.add_argument("--lr_iter", help="number of iterations for LR", default=300)
    parser.add_argument("--n_jobs", help="number of iterations for LR", default=6)
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

    # convert arguments to int
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
        dtype = row["dtype"]
        # description = row["description"]
        description = data_file.split(".")[0]
        log.debug(f'description {description}')

        infile = f'{data_dir}/{data_file}'
        log.info(f"loading file {infile}")
        load_start_time = datetime.now()
        if dtype and len(dtype) > 0:
            df = pd.read_csv(infile, dtype=dtype)
            df = dfu.cast_column_type(df, class_column, "int8")
        else:
            df = pd.read_csv(infile)
        load_end_time = datetime.now()
        load_time_min = round((load_end_time - load_start_time).total_seconds() / 60, 2)
        X_train, X_test, Y_train, Y_test = create_training_data(df, class_column)

        if not args.noknn:
                neigh = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=n_jobs)
                cr.addModel(neigh,
                            X_train,
                            Y_train,
                            X_test,
                            Y_test,
                            file_load_time=load_time_min,
                            name="KNN",
                            description=description,
                            file=data_file,
                            parameters={"n_jobs": n_jobs,
                                        "n_neighbors": neighbors})

        if not args.norn:
                rnc = RadiusNeighborsClassifier(radius=radius, n_jobs=n_jobs)
                cr.addModel(rnc,
                            X_train,
                            Y_train,
                            X_test,
                            Y_test,
                            file_load_time=load_time_min,
                            name="RN",
                            description=description,
                            file=data_file,
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
                            file_load_time=load_time_min,
                            name="LR",
                            description=description,
                            file=data_file,
                            parameters={"n_jobs": n_jobs,
                                        "c": lr_c,
                                        "max_iter": lr_iter} )

        report_df = cr.runNewModels()
    print(report_df.tail())


    log.info("Finished running all models")
    end_time = datetime.now()
    total_time = end_time - start_time
    log.info(f'Total time: {total_time}')
    print(report_df.head())

