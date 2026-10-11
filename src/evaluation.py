# Evaluation helpers shared by the modeling notebooks.
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


def ks_statistic(y_true, p):
    """Kolmogorov-Smirnov statistic: largest gap between score distributions of defaulters and non-defaulters."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    return ks_2samp(p[y_true == 1], p[y_true == 0]).statistic


def evaluate_scores(y_true, p):
    """Ranking metrics from predicted default probabilities."""
    auc = roc_auc_score(y_true, p)
    return {
        "roc_auc": auc,
        "pr_auc": average_precision_score(y_true, p),
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, p),
    }