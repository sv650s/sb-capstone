from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import util.dict_util as du



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


