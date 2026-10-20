"""Classification metrics, written out by hand so the tests can check them line by line."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """cm[i, j] = number of samples of true class i predicted as class j."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum())


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 per class. A class never predicted and never present would give 0/0; it gets 0."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1 = per_class_f1(cm)
    return {
        "accuracy": accuracy(cm),
        "f1_macro": float(f1.mean()),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }
