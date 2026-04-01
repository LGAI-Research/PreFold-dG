import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score


def pcc(pred, true):
    return pearsonr(pred, true).statistic


def srcc(pred, true):
    return spearmanr(pred, true).statistic


def rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())


def mae(pred, true):
    return np.abs(pred - true).mean()


def auroc(pred, true):
    return roc_auc_score(true > 0, pred)


def _optimal_threshold(pred, true):
    fpr, tpr, thresholds = roc_curve(true > 0, pred)
    return thresholds[np.argmax(tpr - fpr)]


def precision(pred, true):
    th = _optimal_threshold(pred, true)
    return precision_score(true < 0, pred < th)


def recall(pred, true):
    th = _optimal_threshold(pred, true)
    return recall_score(true < 0, pred < th)
