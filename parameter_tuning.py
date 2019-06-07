import argparse
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from util.ClassifierRunner import Keys
from util.program_util import Timer
import util.file_util as fu
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from util.dict_util import add_dict_to_dict
from sklearn.externals import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from datetime import datetime

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
TRUE_LIST = ["yes", "Yes", "YES", "y", "True", "true", "TRUE"]
log = logging.getLogger(__name__)
CV_TIME_MIN = "cv_time_min"
MODEL_SAVE_TIME_MIN = "model_save_time_min"
FEATURE_PICKLE_TIME_MIN = "feature_pickle_time_min"


def run_cv(trainer, model_name, parameters, x_train, y_train, x_test, y_test, infile, report, timer, use_random=False):
    log.info(f"Starting to train {model_name}\n\tparameters: {parameters}")
    report["model_name"] = model_name
    if use_random:
        cv = RandomizedSearchCV(estimator=trainer, cv=5, param_distributions=parameters, iid=False)
    else:
        cv = GridSearchCV(estimator=trainer, cv=5, param_grid=parameters, iid=False)
    timer.start_timer(CV_TIME_MIN)
    best_model = cv.fit(x_train, y_train)
    timer.end_timer(CV_TIME_MIN)

    log.info(f'Best Estimatator:\n{best_model.best_estimator_}')
    log.info(f'Best Index:\n{best_model.best_index_}')
    log.info(f'Best Score:\n{best_model.best_score_}')
    log.info(f'Best Params:\n{best_model.best_params_}')

    report[Keys.PARAMETERS] = parameters
    report["best_estimator"] = best_model.best_estimator_
    report["best_index"] = best_model.best_index_
    report["best_score"] = best_model.best_score_
    report["best_params"] = best_model.best_params_

    timer.start_timer(Keys.TRAIN_TIME_MIN)
    model = best_model.fit(x_train, y_train)
    timer.end_timer(Keys.TRAIN_TIME_MIN)

    timer.start_timer(MODEL_SAVE_TIME_MIN)
    _, infile_basename = fu.get_dir_basename(infile)
    joblib_filename = f"models/{datetime.now().strftime(DATE_FORMAT)}-{infile_basename}-{model_name}-{sm_desc}-best.pkl"
    with open(joblib_filename, 'wb') as file:
        joblib.dump(model, file)
    report["model_file"] = joblib_filename
    timer.end_timer(MODEL_SAVE_TIME_MIN)

    timer.start_timer(Keys.PREDICT_TIME_MIN)
    y_predict = model.predict(x_test)
    timer.end_timer(Keys.PREDICT_TIME_MIN)

    c_report = classification_report(y_test, y_predict, output_dict=True)
    report = add_dict_to_dict(report, c_report)
    report.update(timer.get_report())

    log.info(f"Finished training {model_name}\n\tparameters: {parameters}")
    return report


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add a class column ')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    parser.add_argument("--lr_iter", help="logistic regression iterations", default=100)
    parser.add_argument("--n_jobs", help="n_jobs for classifiers", default=-1)
    parser.add_argument("--smote", help="enable smote", action='store_true')
    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)

    config_df = pd.read_csv(args.config_file)
    report_df = pd.DataFrame()
    _, config_basename = fu.get_dir_basename(args.config_file)
    reportfile = f'reports/{config_basename}-report.csv'

    for index, row in config_df.iterrows():
        data_dir = row["data_dir"]
        data_file = row["data_file"]
        class_column = row["class_column"]
        drop_columns = row["drop_columns"]
        config_lrb = row["lrb"]
        run_lrb = config_lrb in TRUE_LIST
        config_cb = row["cb"]
        run_cb = config_cb in TRUE_LIST
        config_lgbm = row["lGBM"]
        run_lGBM = config_lgbm in TRUE_LIST
        infile = f'{data_dir}/{data_file}'
        timer = Timer()

        report = {"file": data_file}

        timer.start_timer(Keys.FILE_LOAD_TIME_MIN)
        df = pd.read_csv(infile)
        timer.end_timer(Keys.FILE_LOAD_TIME_MIN)
        y = df[class_column]

        log.debug(f'class_column {class_column}')
        log.debug(y.head())

        if drop_columns:
            drop_list = drop_columns.replace(" ", "").split(",")
        x = df.drop(drop_list, axis=1)
        x = df.drop(class_column, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

        log.debug(y_train.head())

        timer.start_timer(Keys.SMOTE_TIME_MIN)
        if args.smote:
            sm_desc = "smote"

            log.debug(f'y_train {y_train.shape}')
            log.debug(f'y_train {y_train.head()}')

            grouped_df = y_train.reset_index().groupby("star_rating").count()

            log.debug(f'grouped type: {type(grouped_df)}')
            log.debug(f'grouped: {grouped_df.head()}')
            log.debug(f'grouped: {grouped_df.shape}')

            smote_size = int(round(grouped_df.iloc[0] * 1.66))

            sm = SMOTE(random_state=2, sampling_strategy={1: int(round(grouped_df.iloc[0] * 1.67)),
                                                          2: int(round(grouped_df.iloc[1] * 2.00)),
                                                          3: int(round(grouped_df.iloc[2] * 1.67)),
                                                          4: int(round(grouped_df.iloc[3] * 1.67))}
                       )

            x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

            x_train = pd.DataFrame(x_train_res, columns=x_train.columns)
            y_train = pd.DataFrame(y_train_res, columns=["star_rating"])

            row, col = df.shape

            dist = y_train.reset_index().groupby("star_rating").count()

            log.debug(dist.head())
            _, basename = fu.get_dir_basename(infile)
            dist.to_csv(f'reports/{basename}-smotehist.csv')
        else:
            sm_desc = "nosmote"
        timer.end_timer(Keys.SMOTE_TIME_MIN)

        if run_lGBM:
            model_name = "lGBM"
            # parameters = {"num_leaves": [31, 62, 124],
            #               "min_data_in_leaf": [20, 40, 80],
            #               "max_depth": [4, 8, 16]}
            parameters = {"num_trees": sp_randint(50, 300)}
            trainer = lgb.LGBMClassifier(objective="multiclass",
                                         seed=1)
            report = run_cv(trainer, model_name, parameters, x_train, y_train, x_test, y_test, infile, report, timer,
                            use_random=True)
            report_df = report_df.append(report, ignore_index=True)
            report_df.to_csv(reportfile, index=False)

        if run_cb:
            model_name = "CB"
            # parameters = {"max_depth": [4, 8, 16],
            #               "l2_leaf_reg": [1, 10, 100],
            #               "learning_rate": [0.01, 0.1, 1]}
            parameters = {"iterations": sp_randint(50, 300)}
            trainer = CatBoostClassifier(random_seed=1, loss_function='MultiClass', objective='MultiClass')
            report = run_cv(trainer=trainer, model_name=model_name, parameters=parameters, x_train=x_train,
                            y_train=y_train, x_test=x_test, y_test=y_test, infile=infile, report=report,
                            timer=timer, use_random=True)
            report_df = report_df.append(report, ignore_index=True)
            report_df.to_csv(reportfile, index=False)

        if run_lrb:
            model_name = "LRB100"
            parameters = {'C': [1, 10, 100]}
            trainer = LogisticRegression(random_state=0, solver='lbfgs',
                                         multi_class='auto',
                                         class_weight='balanced',
                                         max_iter=args.lr_iter, n_jobs=args.n_jobs,
                                         verbose=1)
            report = run_cv(trainer, model_name, parameters, x_train, y_train, x_test, y_test, infile, report, timer)
            report_df = report_df.append(report, ignore_index=True)
            report_df.to_csv(reportfile, index=False)
