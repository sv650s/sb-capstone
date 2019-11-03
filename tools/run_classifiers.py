import sys
sys.path.append('../')

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from util.model_util import Model
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import pandas as pd
import logging
import util.file_util as fu
import util.df_util as dfu
import lightgbm as lgb
import gc
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from util.program_util import TimedProgram, ConfigFileBasedProgram
from util.time_util import Keys, Status

# configure logger so we can see output from the classes
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TRUE_LIST = ["yes", "Yes", "YES", "y", "True", "true", "TRUE"]
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
log = logging.getLogger(__name__)

RSTATE=1

class TimedClassifier(TimedProgram):
    """
    This represents a model and one run of the model. Pertinent information will be timed if model is wrapped in
    this class

    Configuration for this class will be based on a row in pandas dataframe.

    Current the following columns are expected in the configuration file (see template-run_classifier.csv for sample):
        data_dir - directory of data file
        data_file - name of the data file with features to load
        model_name - name you want to name the current model. will be used to create the model
        class_column - this column will be extracted from data file to be used as labels for our model
        drop_columns - comma separated list of columns to drop (optional)
        smote - Yes/No - indicate whether we should use SMOTE to synthesize features to balance classes
        dtype - explicitly specify this if you want to cast all columns to a certain data time (optional)
    """

    def __init__(self, index, config_df, report=None, args=None):
        super(TimedClassifier, self).__init__(index, config_df, report, args)
        self.n_jobs = int(self.args.n_jobs)
        self.lr_iter = int(self.args.lr_iter)
        self.neighbors = int(self.args.neighbors)
        self.radius = int(self.args.radius)
        self.lr_c = int(self.args.lr_c)
        self.epochs = int(self.args.epochs)

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

        return train_test_split(x, y, random_state=RSTATE, stratify=y)

    def execute(self):

        data_file = self.get_config("data_file")
        class_column = self.get_config("class_column")
        drop_columns = self.get_config("drop_columns")
        dtype = self.get_config("dtype")
        sampling = self.get_config("sampling")

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
        if sampling is not None:
            ## if we want to over sample or under sample
            log.debug(f'Y_train {Y_train.shape}')
            log.debug(f'Y_train {Y_train.head()}')

            grouped_df = Y_train.reset_index().groupby("star_rating").count()

            log.debug(f'grouped type: {type(grouped_df)}')
            log.debug(f'grouped: {grouped_df.head()}')
            log.debug(f'grouped: {grouped_df.shape}')

            if sampling == "smote":
                sm_desc = sampling
                sampler = SMOTE(random_state=RSTATE, sampling_strategy='not majority')
            elif sampling == "random_under_sampling":
                sm_desc = sampling
                sampler = RandomUnderSampler(random_state=RSTATE, replacement=True)
            elif sampling == "nearmiss-2":
                sm_desc = sampling
                sampler = NearMiss(random_state=RSTATE, sampling_strategy='not minority', version=2, n_jobs=self.n_jobs)
            else:
                raise Exception(f"Sampling method not supported: {sampling}")

            X_train_res, Y_train_res = sampler.fit_resample(X_train, Y_train.ravel())

            X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
            Y_train = pd.DataFrame(Y_train_res, columns=["star_rating"])

            row, col = df.shape

            dist = Y_train.reset_index().groupby("star_rating").count()

            log.debug(dist.head())
            _, basename = fu.get_dir_basename(infile)
            dist.to_csv(f'../reports/{basename}-histogram-{sampling}.csv')
        else:
            sm_desc = "sampling_none"
        self.stop_timer(Keys.SMOTE_TIME_MIN)

        classifier = None
        if model_name == "DT":
            classifier = DecisionTreeClassifier(random_state=RSTATE)
            parameters = {}

        elif model_name == "DTB":
            classifier = DecisionTreeClassifier(random_state=RSTATE, class_weight='balanced')
            parameters = {"class_weight": "balanced"}

        elif model_name == "RF":
            classifier = RandomForestClassifier(random_state=RSTATE, n_jobs=self.n_jobs, verbose=1)
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
            classifier = LogisticRegression(random_state=RSTATE, solver='lbfgs',
                                            multi_class='auto',
                                            max_iter=self.lr_iter, n_jobs=self.n_jobs, C=self.lr_c,
                                            verbose=1)
            parameters = {"n_jobs": self.n_jobs,
                          "c": self.lr_c,
                          "max_iter": self.lr_iter,
                          "verbose": 1}

        elif model_name == "LRB":
            classifier = LogisticRegression(random_state=RSTATE, solver='lbfgs',
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



        if classifier is not None:
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
        else:
            log.error(f"Unable to create classifier for {model_name}")


if __name__ == "__main__":
    program = ConfigFileBasedProgram("Run classifiers no feature files", TimedClassifier)
    program.add_argument("--noreport", help="do not generate report", action='store_true')
    program.add_argument("--lr_iter", help="number of iterations for LR. default 100", default=100)
    program.add_argument("--n_jobs", help="number of cores to use. default -1", default=-1)
    program.add_argument("--neighbors", help="number of neighbors for KNN", default=5)
    program.add_argument("--radius", help="radius for radius neighbor classification. default 30", default=30)
    program.add_argument("--lr_c", help="c parameter for LR. default 1.0", default=1.0)
    program.add_argument("--epochs", help="epoch for deep learning. Default 1", default=1)
    program.main()
