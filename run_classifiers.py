import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from models.ClassifierRunner import ClassifierRunner
from datetime import datetime
import pandas as pd
import logging
import argparse
import util.file_util as fu
import util.df_util as dfu
import lightgbm as lgb
import gc
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# configure logger so we can see output from the classes
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TRUE_LIST = ["yes", "Yes", "YES", "y", "True", "true", "TRUE"]
LOG_FORMAT='%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)

def create_training_data(x:pd.DataFrame, class_column:str, drop_columns:str = None):
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

    if drop_columns:
        drop_list = drop_columns.replace(" ", "").split(",")
        log.info(f"Dropping columns from features {drop_list}")
        x.drop(drop_list, axis=1, inplace=True)

    log.info(f'shape of x: {x.shape}')
    log.info(f'shape of y: {y.shape}')

    return train_test_split(x, y, random_state=1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add a class column ')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    parser.add_argument("--noreport", help="do not generate report", action='store_true')
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
        drop_columns = row["drop_columns"]
        if pd.notnull(row["dtype"]):
            dtype = np.dtype(row["dtype"])
        else:
            dtype = None

        # let's get the parameters to figure out which training models we need to run
        run_knn = row["knn"] in TRUE_LIST
        run_rn = row["rn"] in TRUE_LIST
        run_lr = row["lr"] in TRUE_LIST
        run_lrb = row["lrb"] in TRUE_LIST
        run_rf = row["rf"] in TRUE_LIST
        run_gb = row["gb"] in TRUE_LIST
        run_lGBM = row["lGBM"] in TRUE_LIST
        run_xgb = row["xgb"] in TRUE_LIST
        run_cb = row["cb"] in TRUE_LIST

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

        X_train, X_test, Y_train, Y_test = create_training_data(df, class_column, drop_columns)

        if run_rf:
            rf = RandomForestClassifier(random_state=1, n_jobs=n_jobs, verbose=1)
            cr.addModel(rf,
                        X_train,
                        Y_train,
                        X_test,
                        Y_test,
                        file_load_time=load_time_min,
                        name="RF",
                        description=description,
                        file=data_file,
                        parameters={"n_jobs": n_jobs, "verbose": 1})

        if run_gb:
            gb = GradientBoostingClassifier(verbose=1)
            cr.addModel(gb,
                        X_train,
                        Y_train,
                        X_test,
                        Y_test,
                        file_load_time=load_time_min,
                        name="GB",
                        description=description,
                        file=data_file,
                        parameters={"verbose": 1})

        if run_lGBM:
            gb = lgb.LGBMClassifier(objective="multiclass", num_threads=2,
                                    seed=1)
            cr.addModel(gb,
                        X_train,
                        Y_train,
                        X_test,
                        Y_test,
                        file_load_time=load_time_min,
                        name="lGBM",
                        description=description,
                        file=data_file,
                        parameters={"objective": "multiclass", "num_threads": 2, "seed": 1})
        if run_xgb:
            xgb = XGBClassifier(n_jobs=n_jobs, verbosity=1, seed=1)
            cr.addModel(xgb,
                        X_train,
                        Y_train,
                        X_test,
                        Y_test,
                        file_load_time=load_time_min,
                        name="XGB",
                        description=description,
                        file=data_file,
                        parameters={"n_jobs": n_jobs, "verbosity": 1, "seed": 1})

        if run_cb:
            cb = CatBoostClassifier(logging_level="Info", random_seed=1)
            cr.addModel(cb,
                        X_train,
                        Y_train,
                        X_test,
                        Y_test,
                        file_load_time=load_time_min,
                        name="CB",
                        description=description,
                        file=data_file,
                        parameters={"logging_level": "Info", "random_seed": 1})




        if run_lr:
            lr = LogisticRegression(random_state=0, solver='lbfgs',
                                    multi_class='auto',
                                    max_iter=lr_iter, n_jobs=n_jobs, C=lr_c,
                                    verbose=1)
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
                                    "max_iter": lr_iter,
                                    "verbose": 1} )

        if run_lrb:
            lrb = LogisticRegression(random_state=0, solver='lbfgs',
                                    multi_class='auto',
                                     class_weight='balanced',
                                    max_iter=lr_iter, n_jobs=n_jobs, C=lr_c,
                                    verbose=1)
            cr.addModel(lr,
                        X_train,
                        Y_train,
                        X_test,
                        Y_test,
                        file_load_time=load_time_min,
                        name="LRB",
                        description=description,
                        file=data_file,
                        parameters={"n_jobs": n_jobs,
                                    "c": lr_c,
                                    "class_weight": 'balanced',
                                    "max_iter": lr_iter,
                                    "verbose": 1} )

        if run_knn:
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

        if run_rn:
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

        report_df = cr.runNewModels()

        config_df.loc[index, "status"] = "success"
        config_df.loc[index, "status_date"] = datetime.now().strftime(TIME_FORMAT)
        config_df.to_csv(args.config_file, index=False)


        gc.collect()
    print(report_df.tail())


    log.info("Finished running all models")
    end_time = datetime.now()
    total_time = end_time - start_time
    log.info(f'Total time: {total_time}')
    print(report_df.head())

