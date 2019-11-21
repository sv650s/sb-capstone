

RSTATE = 1


class Keys(object):
    TRAIN_EXAMPLES = "train_examples"
    TRAIN_FEATURES = "train_features"
    TEST_EXAMPLES = "test_examples"
    TEST_FEATURES = "test_features"
    TRAIN_TIME_MIN = "train_time_min"
    SCORE_TIME_MIN = "score_time_min"
    PREDICT_TIME_MIN = "predict_time_min"
    SMOTE_TIME_MIN = "smote_time_min"
    VECTORIZER_TIME_MIN = "vectorizer_time_min"
    LDA_TIME_MIN = "lda_time_min"
    FILE_LOAD_TIME_MIN = "file_load_time_min"
    TOTAL_TIME_MIN = "total_time_min"
    STATUS = "status"
    STATUS_DATE = "status_date"
    MESSAGE = "message"
    MODEL_NAME = "model_name"
    MODEL_SAVE_TIME_MIN = "model_save_time_min"
    MODEL_FILE = "model_file"
    MODEL = "model"
    FILE = "file"
    DESCRIPTION = "description"
    ROC_AUC = "roc_auc"
    PARAMETERS = "param"
    TRAIN_X = "X_train"
    TRAIN_Y = "Y_train"
    TEST_X = "X_test"
    TEST_Y = "Y_test"
    TIMER = "timer"
    CM = "confusion_matrix"
    CR = "classification_report"
    CONFIG_FILE = "config_file"


class Status(object):
    FAILED = "failed"
    SUCCESS = "success"
    NEW = "new"