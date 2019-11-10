import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def calculate_roc_auc(y_test: np.ndarray, y_predict: pd.DataFrame):
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

    :param y_test: test labels. should be np array or dataframe
    :param y_predict: test predictions - this should either be an np.ndarray or DataFrame
    :return:
    """
    if isinstance(y_predict, np.ndarray):
        y_predict =  pd.DataFrame(y_predict)

    print(y_test.shape)
    n_classes = y_test.shape[1]

    # to compute the micro RUC/AUC, we need binariized labels (y_test) and probability of predictions y_predict_df

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in np.arange(0, 5):
        fpr_ndarray, tpr_ndarray, _ = roc_curve(y_test[:, i], y_predict[i].to_list())
        fpr[str(i)] = fpr_ndarray.tolist()
        tpr[str(i)] = tpr_ndarray.tolist()
        roc_auc[f'auc_{i + 1}'] = auc(fpr[str(i)], tpr[str(i)]).tolist()

    # Compute micro-average ROC curve and ROC area
    fpr_ndarray, tpr_ndarray, _ = roc_curve(y_test.ravel(), y_predict.values.ravel())
    fpr["micro"] = fpr_ndarray.tolist()
    tpr["micro"] = tpr_ndarray.tolist()
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"]).tolist()

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[str(i)] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[str(i)], tpr[str(i)])

    # Finally aveage it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr.tolist()
    tpr["macro"] = mean_tpr.tolist()
    roc_auc["auc_macro"] = auc(fpr["macro"], tpr["macro"]).tolist()

    return roc_auc, fpr, tpr