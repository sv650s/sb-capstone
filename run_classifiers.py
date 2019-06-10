from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from util.ClassifierRunner import Model
from imblearn.over_sampling import SMOTE
import pandas as pd
import logging
import util.file_util as fu
import util.df_util as dfu
import lightgbm as lgb
import gc
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from util.ConfigBasedProgram import TimedProgram, ConfigBasedProgram
from util.program_util import Keys, Status

# configure logger so we can see output from the classes
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TRUE_LIST = ["yes", "Yes", "YES", "y", "True", "true", "TRUE"]
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)


class RunClassifiers(TimedProgram):

    def __init__(self, index, config_df, report=None, args=None):
        super(RunClassifiers, self).__init__(index, config_df, report, args)
        self.n_jobs = int(self.args.n_jobs)
        self.lr_iter = int(self.args.lr_iter)
        self.neighbors = int(self.args.neighbors)
        self.radius = int(self.args.radius)
        self.lr_c = int(self.args.lr_c)

    def create_training_data(self, x: pd.DataFrame, class_column: str, drop_columns: str = None):
        """
        Take dataframe and:
        1. split between features and predictions
        2. create test and training sets
        :param x:
        :param class_column:
        :param drop_columns:
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

        return train_test_split(x, y, random_state=1, stratify=y)

    def execute(self):

        data_file = self.get_config("data_file")
        class_column = self.get_config("class_column")
        drop_columns = self.get_config("drop_columns")
        dtype = self.get_config("dtype")

        # let's get the parameters to figure out which training models we need to run
        model_name = self.get_config("model_name")

        # description = row["description"]
        description = data_file.split(".")[0]
        log.debug(f'description {description}')

        infile = self.get_infile()
        log.info(f"loading file {infile}")
        self.start_timer(Keys.FILE_LOAD_TIME_MIN)
        if dtype and len(dtype) > 0:
            df = pd.read_csv(infile, dtype=dtype)
            df = dfu.cast_column_type(df, class_column, "int8")
        else:
            df = pd.read_csv(infile)
        self.stop_timer(Keys.FILE_LOAD_TIME_MIN)

        X_train, X_test, Y_train, Y_test = self.create_training_data(df, class_column, drop_columns)

        # put these here so our columns are not messed up
        self.start_timer(Keys.SMOTE_TIME_MIN)
        if self.args.smote:
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
        self.stop_timer(Keys.SMOTE_TIME_MIN)

        if model_name == "RF":
            classifier = RandomForestClassifier(random_state=1, n_jobs=self.n_jobs, verbose=1)
            parameters = {"n_jobs": self.n_jobs, "verbose": 1}

        elif model_name == "GB":
            classifier = GradientBoostingClassifier(verbose=1)
            parameters = {"verbose": 1}

        elif model_name == "lGBM":
            classifier = lgb.LGBMClassifier(objective="multiclass", num_threads=2,
                                            seed=1)
            parameters = {"objective": "multiclass", "num_threads": 2, "seed": 1}

        elif model_name == "XGB":
            classifier = XGBClassifier(n_jobs=self.n_jobs, verbosity=1, seed=1)
            parameters = {"n_jobs": self.n_jobs, "verbosity": 1, "seed": 1}

        elif model_name == "CB":
            classifier = CatBoostClassifier(random_seed=1)
            parameters = {"random_seed": 1}

        elif model_name == "LR":
            classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                            multi_class='auto',
                                            max_iter=self.lr_iter, n_jobs=self.n_jobs, C=self.lr_c,
                                            verbose=1)
            parameters = {"n_jobs": self.n_jobs,
                          "c": self.lr_c,
                          "max_iter": self.lr_iter,
                          "verbose": 1}

        elif model_name == "LRB100":
            classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                            multi_class='auto',
                                            class_weight='balanced',
                                            max_iter=self.lr_iter, n_jobs=self.n_jobs, C=self.lr_c,
                                            verbose=1)
            parameters = {"n_jobs": self.n_jobs,
                          "c": self.lr_c,
                          "class_weight": 'balanced',
                          "max_iter": self.lr_iter,
                          "verbose": 1}

        elif model_name == "KNN":
            classifier = KNeighborsClassifier(n_neighbors=self.neighbors, n_jobs=self.n_jobs)
            parameters = {"n_jobs": self.n_jobs,
                          "n_neighbors": self.neighbors}

        elif model_name == "RN":
            classifier = RadiusNeighborsClassifier(radius=self.radius, n_jobs=self.n_jobs)
            parameters = {"n_jobs": self.n_jobs,
                          "radius": self.radius}

        model = Model(classifier,
                      X_train,
                      Y_train,
                      X_test,
                      Y_test,
                      name=model_name,
                      class_column=class_column,
                      description=f'{description}-{sm_desc}',
                      file=data_file,
                      parameters=parameters
                      )
        report_dict, _ = model.run()
        self.report.add_dict(report_dict)


if __name__ == "__main__":
    program = ConfigBasedProgram("Run classifiers no feature files", RunClassifiers)
    program.add_argument("--noreport", help="do not generate report", action='store_true')
    program.add_argument("--lr_iter", help="number of iterations for LR. default 100", default=100)
    program.add_argument("--n_jobs", help="number of cores to use. default -1", default=-1)
    program.add_argument("--neighbors", help="number of neighbors for KNN", default=5)
    program.add_argument("--radius", help="radius for radius neighbor classification. default 30", default=30)
    program.add_argument("-s", "--smote", help="if yes, will run the data with SMOTE. default False",
                         action='store_true')
    program.add_argument("--lr_c", help="c parameter for LR. default 1.0", default=1.0)
    program.main()
