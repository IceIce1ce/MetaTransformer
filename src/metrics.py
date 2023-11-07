import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, classification_report
import itertools

def flat_list(list_):
    if isinstance(list_, (np.ndarray, np.generic)):
        return list_
    return list(itertools.chain(*list_))

def evaluation(outputs, targets, **kwargs):
    preds = np.array(flat_list(outputs))
    gts = np.array(flat_list(targets))
    F1_mean, F1_one = f1_mean(gts, preds)
    f_auc = roc_auc_score(gts, preds)
    pr_auc = average_precision_score(gts, preds)
    accuracy = accuracy_score(gts, preds > 0.5)
    report = classification_report(gts, preds > 0.5, target_names=['normal', 'anomaly'])
    print("f-AUC = %.5f" % (f_auc))
    print("PR-AUC = %.5f" % (pr_auc))
    print("F1-Score = %.5f" % (F1_one))
    print("F1-Mean  = %.5f" % (F1_mean))
    print("Accuracy = %.5f" % (accuracy))
    print()
    print(report)

def f1_mean(gts, preds):
    F1_one = f1_score(gts, preds > 0.5)
    F1_zero = f1_score((gts.astype('bool')==False).astype('long'), preds <= 0.5)
    F1_mean = 2 * (F1_one * F1_zero) / (F1_one + F1_zero)
    return F1_mean, F1_one