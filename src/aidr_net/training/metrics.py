"""Binary metrics; rank metrics include score ties and single-class handling."""
import numpy as np


def binary_metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = p >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    tn, fn = int(((y == 0) & ~pred).sum()), int(((y == 1) & ~pred).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    result = {"accuracy": (tp + tn) / max(len(y), 1), "precision": precision, "recall": recall,
              "f1": 2 * precision * recall / max(precision + recall, 1e-12),
              "fpr": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
              "auroc": None, "average_precision": None}
    if len(np.unique(y)) == 2:
        order = np.argsort(-p, kind="stable")
        sorted_y, sorted_p = y[order], p[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y)-1]
        tps = np.r_[0, np.cumsum(sorted_y)[ends]]
        fps = np.r_[0, ends + 1 - np.cumsum(sorted_y)[ends]]
        tpr, fpr = tps / (y == 1).sum(), fps / (y == 0).sum()
        result["auroc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        result["average_precision"] = float(np.sum(np.diff(tpr) * tps[1:] / (tps[1:] + fps[1:])))
    return result
