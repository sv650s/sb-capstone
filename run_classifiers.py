import numpy as np
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from util.ClassifierRunner import ClassifierRunner, Keys, Timer, Model
from imblearn.over_sampling import SMOTE
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
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)


def create_training_data(x: pd.DataFrame, class_column: str, drop_columns: str = None):
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
    parser.add_argument("--lr_iter", help="number of iterations for LR", default=100)
    parser.add_argument("--n_jobs", help="number of iterations for LR", default=-1)
    parser.add_argument("--neighbors", help="number of neighbors for KNN", default=5)
    parser.add_argument("--radius", help="radius for radius neighbor classification", default=30)
    parser.add_argument("-s", "--smote", help="if yes, will run the data with SMOTE", action='store_true')
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

        timer = Timer()

        infile = f'{data_dir}/{data_file}'
        log.info(f"loading file {infile}")
        load_start_time = datetime.now()
        timer.start_timer(Keys.FILE_LOAD_TIME_MIN)
        if dtype and len(dtype) > 0:
            df = pd.read_csv(infile, dtype=dtype)
            df = dfu.cast_column_type(df, class_column, "int8")
        else:
            df = pd.read_csv(infile)
        timer.end_timer(Keys.FILE_LOAD_TIME_MIN)
        load_end_time = datetime.now()
        load_time_min = round((load_end_time - load_start_time).total_seconds() / 60, 2)

        X_train, X_test, Y_train, Y_test = create_training_data(df, class_column, drop_columns)

        # put these here so our columns are not messed up
        timer.start_timer(Keys.SMOTE_TIME_MIN)
        if args.smote:
            sm_desc = "smote"

            log.debug(f'Y_train {Y_train.shape}')
            log.debug(f'Y_train {Y_train.head()}')

            grouped_df = Y_train.reset_index().groupby("star_rating").count()

            log.debug(f'grouped type: {type(grouped_df)}')
            log.debug(f'grouped: {grouped_df.head()}')
            log.debug(f'grouped: {grouped_df.shape}')

            sm = SMOTE(random_state=2, sampling_strategy={1: int(round(grouped_df.iloc[0] * 1.67)),
                                                          2: int(round(grouped_df.iloc[1] * 2.00)),
                                                          3: int(round(grouped_df.iloc[2] * 1.67)),
                                                          4: int(round(grouped_df.iloc[3] * 1.67))}
                       )

            X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train.ravel())

            X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
            Y_train = pd.DataFrame(Y_train_res, columns=["star_rating"])

            row, col = df.shape

            dist = Y_train.reset_index().groupby("star_rating").count()

            log.debug(dist.head())
            _, basename = fu.get_dir_basename(infile)
            dist.to_csv(f'reports/{basename}-smotehist.csv')
        else:
            sm_desc = "nosmote"
        timer.end_timer(Keys.SMOTE_TIME_MIN)

        if run_rf:
            rf = RandomForestClassifier(random_state=1, n_jobs=n_jobs, verbose=1)
            model = Model(model=rf,
                          x_train=X_train,
                          y_train=Y_train,
                          x_test=X_test,
                          y_test=Y_test,
                          name="RF",
                          timer=timer,
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          parameters={"n_jobs": n_jobs, "verbose": 1}
                          )
            cr.addModel(model)

        if run_gb:
            gb = GradientBoostingClassifier(verbose=1)
            model = Model(gb,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name="GB",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          )
            cr.addModel(model)

        if run_lGBM:
            gb = lgb.LGBMClassifier(objective="multiclass", num_threads=2,
                                    seed=1)
            model = Model(gb,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name="lGBM",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          )
            cr.addModel(model)

        if run_xgb:
            xgb = XGBClassifier(n_jobs=n_jobs, verbosity=1, seed=1)
            model = Model(xgb,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name="XGB",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          parameters={"n_jobs": n_jobs, "verbosity": 1, "seed": 1}
                          )
            cr.addModel(model)

        if run_cb:
            cb = CatBoostClassifier(random_seed=1)
            model = Model(cb,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name="CB",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          parameters={"random_seed": 1}
                          )
            cr.addModel(model)

        if run_lr:
            lr = LogisticRegression(random_state=0, solver='lbfgs',
                                    multi_class='auto',
                                    max_iter=lr_iter, n_jobs=n_jobs, C=lr_c,
                                    verbose=1)
            model = Model(lr,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name=f"LR{lr_iter}",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          parameters={"n_jobs": n_jobs,
                                      "c": lr_c,
                                      "max_iter": lr_iter,
                                      "verbose": 1}
                          )
            cr.addModel(model)

        if run_lrb:
            lrb = LogisticRegression(random_state=0, solver='lbfgs',
                                     multi_class='auto',
                                     class_weight='balanced',
                                     max_iter=lr_iter, n_jobs=n_jobs, C=lr_c,
                                     verbose=1)
            model = Model(lrb,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name=f"LRB{lr_iter}",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          parameters={"n_jobs": n_jobs,
                                      "c": lr_c,
                                      "class_weight": 'balanced',
                                      "max_iter": lr_iter,
                                      "verbose": 1}
                          )
            cr.addModel(model)

        if run_knn:
            neigh = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=n_jobs)
            model = Model(neigh,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name="KNN",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          parameters={"n_jobs": n_jobs,
                                      "n_neighbors": neighbors}
                          )
            cr.addModel(model)

        if run_rn:
            rnc = RadiusNeighborsClassifier(radius=radius, n_jobs=n_jobs)
            model = Model(rnc,
                          X_train,
                          Y_train,
                          X_test,
                          Y_test,
                          timer=timer,
                          name="RN",
                          description=f'{description}-{sm_desc}',
                          file=data_file,
                          parameters={"n_jobs": n_jobs,
                                      "radius": radius}
                          )
            cr.addModel(model)

        report_df = cr.run_models()

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
