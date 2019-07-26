from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import util.dict_util as du
import util.file_util as fu


DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'



def calculate_roc_auc(y_test: np.ndarray, y_score: pd.DataFrame):
    """
    Caculates the false positive rate, true positive rate, and roc_auc
    Return:
      roc_auc - dictinary of AUC values there will be 1 auc per class then a macro
          and micro for the overall model
          keys: auc_0 to auc_{n_classes} + micro and macro
      fpr - dictionary of FPR, each value will be a n_classes x ? interpolated array
          so you can plt ROC curve for each class
          keys: 0 to {n_classes} + micro and macro
      tpr - dictionary of TPR, each value will be a n_classes x ? interpolated array
          so used to plt ROC curve for each class
          keys: 0 to {n_classes} + micro and macro

    :param y_test: test labels
    :param y_score: test predictions - this should either be an np.ndarray or DataFrame
    :return:
    """
    if isinstance(y_score, np.ndarray):
        y_score =  pd.DataFrame(y_score)


    n_classes = y_test.shape[1]

    # to compute the micro RUC/AUC, we need binariized labels (y_test) and probability of predictions y_predict_df

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in np.arange(0, 5):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[i].to_list())
        roc_auc[f'auc_{i + 1}'] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.values.ravel())
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally aveage it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["auc_macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc, fpr, tpr


def unencode(input):
    """
    unencode input
    input is assumed to be probability returned by model - ie, columns are the classes, values in each row is the probability of each class
    unencoded output would be a 1D array with each element containing the class index with highest probability

    :param input: np array or pandas DataFrame - should be 2D
    :return:
    """
    if isinstance(input, np.ndarray):
        input_df = pd.DataFrame(input)
    elif isinstance(input, pd.DataFrame):
        input_df = input

    return [row.idxmax() + 1 for index, row in input_df.iterrows()]


def preprocess_file(data_df, feature_column, label_column, keep_percentile):
    """
    Preprocessing data file and create the right inputs for Keras models
        Features:
        * tokenize
        * pad features into sequence
        Labels:
        * one hot encoder

        split between training and testing

    :param data_df:
    :param feature_column:
    :param review_column:
    :param keep_percentile: percentile of feature length to keep - all features will be padded to this length
    :return:
    """
    labels = data_df[label_column]
    features = data_df[feature_column]

    print("One hot enocde label data...")
    y = OneHotEncoder().fit_transform(labels.values.reshape(len(labels), 1)).toarray()

    # split our data into train and test sets
    print("Splitting data into training and test sets...")
    features_train, features_test, y_train, y_test = train_test_split(features, y, random_state=1)

    # Pre-process our features (review body)
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(features_train)
    # tokenize both our training and test data
    train_sequences = t.texts_to_sequences(features_train)
    test_sequences = t.texts_to_sequences(features_test)

    print("Vocabulary size={}".format(len(t.word_counts)))
    print("Number of Documents={}".format(t.document_count))

    # figure out 99% percentile for our max sequence length
    data_df["feature_length"] = data_df.review_body.apply(lambda x: len(x.split()))
    max_sequence_length = int(data_df.feature_length.quantile([keep_percentile]).values[0])
    print(f'Max Sequence Length: {max_sequence_length}')

    # pad our reviews to the max sequence length
    X_train = sequence.pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test = sequence.pad_sequences(test_sequences, maxlen=max_sequence_length)

    return X_train, X_test, y_train, y_test, t, max_sequence_length


class ModelEvaluator(object):

    def __init__(self, name, model, network_history):
        self.name = name
        self.model = model
        self.network_history = network_history

    def evaluate(self, X_test, y_test):
        print("Running model.evaluate...")
        self.scores = self.model.evaluate(X_test, y_test, verbose=1)

        print("Running model.predict...")
        # this is a 2D array with each column as probabilyt of that class
        self.y_predict = self.model.predict(X_test)

        print("Unencode predictions...")
        # 1D array with class index as each value
        y_predict_unencoded = unencode(self.y_predict)
        y_test_unencoded = unencode(y_test)


        print("Generating confusion matrix...")
        self.confusion_matrix = confusion_matrix(y_test_unencoded, y_predict_unencoded)

        print("Calculating ROC AUC...")
        self.roc_auc, self.fpr, self.tpr = calculate_roc_auc(y_test, self.y_predict)

        print("Getting classification report...")
        self.classification_report = classification_report(y_test_unencoded, y_predict_unencoded)
        # classification report dictioonary
        self.crd = classification_report(y_test_unencoded, y_predict_unencoded, output_dict=True)


