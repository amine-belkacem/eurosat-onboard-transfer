"""Metrics and result aggregation, against values computed by hand."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from eot.metrics import accuracy, confusion_matrix, per_class_f1, summarize
from eot.report import aggregate, load_runs


def check(got, expected, what=""):
    assert got == expected, f"ECHEC {what}: obtenu {got}, attendu {expected}"


def check_close(got, expected, what="", tol=1e-9):
    assert abs(got - expected) <= tol, f"ECHEC {what}: obtenu {got}, attendu {expected}"


# y_true = 0 0 1 1 2 2
# y_pred = 0 1 1 1 2 0
#             true 0: one right, one predicted 1      -> row [1, 1, 0]
#             true 1: both right                       -> row [0, 2, 0]
#             true 2: one right, one predicted 0      -> row [1, 0, 1]
Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])


def test_confusion_matrix():
    check(confusion_matrix(Y_TRUE, Y_PRED, 3).tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 1]], "matrice")


def test_accuracy():
    # 4 correct out of 6
    check_close(accuracy(confusion_matrix(Y_TRUE, Y_PRED, 3)), 4 / 6, "accuracy")


def test_per_class_f1():
    # class 0: tp=1 fp=1 fn=1 -> 2/(2+1+1) = 0.5
    # class 1: tp=2 fp=1 fn=0 -> 4/(4+1+0) = 0.8
    # class 2: tp=1 fp=0 fn=1 -> 2/(2+0+1) = 2/3
    f1 = per_class_f1(confusion_matrix(Y_TRUE, Y_PRED, 3))
    for got, exp, c in zip(f1, [0.5, 0.8, 2 / 3], range(3)):
        check_close(got, exp, f"F1 classe {c}")


def test_macro_f1():
    # (0.5 + 0.8 + 2/3) / 3 = 0.655555...
    check_close(summarize(Y_TRUE, Y_PRED, 3)["f1_macro"], (0.5 + 0.8 + 2 / 3) / 3, "F1 macro")


def test_f1_absent_class_is_zero():
    # class 2 never appears and is never predicted: 0/0 -> 0, not NaN
    f1 = per_class_f1(confusion_matrix(np.array([0, 1]), np.array([0, 1]), 3))
    check(f1.tolist(), [1.0, 1.0, 0.0], "F1 classe absente")


def test_confusion_shape_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0]), 2)


def _write_run(directory, config, pct, seed, acc, time_s, device="Tesla T4", feature_time_s=0.0):
    directory.mkdir(parents=True, exist_ok=True)
    run = {"config": config, "model": "m", "mode": "finetune", "pct": pct, "seed": seed,
           "n_train": 10, "best_val_acc": acc, "test_accuracy": acc, "test_f1_macro": acc,
           "train_time_total_s": time_s, "time_to_best_s": time_s, "feature_time_s": feature_time_s,
           "best_step": 1, "steps_done": 1, "stopped_early": False, "n_params": 1,
           "n_trainable": 1, "device": device}
    (directory / f"pct{pct:03d}_seed{seed}.json").write_text(json.dumps(run))


def test_aggregate_mean_std(tmp_path):
    # accuracies 0.90, 0.92, 0.94: mean 0.92
    # sample std = sqrt((0.02^2 + 0 + 0.02^2) / (3 - 1)) = sqrt(0.0004) = 0.02
    for seed, acc in enumerate([0.90, 0.92, 0.94]):
        _write_run(tmp_path / "cfg", "cfg", 100, seed, acc, 60.0 * (seed + 1))
    _write_run(tmp_path / "_smoke" / "cfg", "cfg", 100, 9, 0.0, 1.0)  # must be ignored
    agg = aggregate(load_runs(tmp_path))
    row = agg.iloc[0]
    check(int(row["n_seeds"]), 3, "nombre de graines (smoke ignore)")
    check_close(row["test_accuracy_mean"], 0.92, "moyenne", 1e-12)
    check_close(row["test_accuracy_std"], 0.02, "ecart-type", 1e-12)
    # times 60, 120, 180 s: mean 120, std sqrt((3600+0+3600)/2) = 60
    check_close(row["train_time_total_s_mean"], 120.0, "temps moyen", 1e-9)
    check_close(row["train_time_total_s_std"], 60.0, "ecart-type temps", 1e-9)


def test_aggregate_refuses_mixed_devices(tmp_path):
    _write_run(tmp_path / "cfg", "cfg", 100, 0, 0.9, 10.0, device="Tesla T4")
    _write_run(tmp_path / "cfg", "cfg", 100, 1, 0.9, 10.0, device="Tesla P100-PCIE-16GB")
    with pytest.raises(ValueError):
        aggregate(load_runs(tmp_path))


def test_time_to_best_includes_feature_extraction(tmp_path):
    # probe: 40 s of feature extraction + best head after 5 s -> 45 s
    _write_run(tmp_path / "probe", "probe", 100, 0, 0.9, 5.0, feature_time_s=40.0)
    check_close(load_runs(tmp_path)["time_to_best_total_s"].iloc[0], 45.0, "temps jusqu'au meilleur (sonde)")
