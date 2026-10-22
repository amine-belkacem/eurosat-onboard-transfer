"""Aggregate per-run JSON files into mean / std tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUN_COLUMNS = ["config", "model", "mode", "pct", "seed", "n_train", "best_val_acc", "test_accuracy",
               "test_f1_macro", "train_time_total_s", "time_to_best_s", "feature_time_s",
               "best_step", "steps_done", "stopped_early", "n_params", "n_trainable", "device"]
METRICS = ["test_accuracy", "test_f1_macro", "train_time_total_s", "time_to_best_total_s"]


def load_runs(results_dir: Path) -> pd.DataFrame:
    """One row per run, ignoring smoke runs."""
    rows = []
    for path in sorted(results_dir.glob("*/pct*_seed*.json")):
        if path.parent.name.startswith("_"):
            continue
        run = json.loads(path.read_text())
        rows.append({k: run.get(k) for k in RUN_COLUMNS})
    if not rows:
        raise FileNotFoundError(f"no result files under {results_dir}")
    runs = pd.DataFrame(rows)
    # time_to_best_s is measured inside the training loop. For the probe the
    # feature extraction happens before it and must be added, otherwise the
    # probe would look faster than it is.
    runs["time_to_best_total_s"] = runs["feature_time_s"] + runs["time_to_best_s"]
    return runs


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std (ddof=1) over seeds for each config and fraction.

    Groups with mixed devices are refused: their times would not be comparable.
    """
    devices = runs.groupby(["config", "pct"])["device"].nunique()
    if (devices > 1).any():
        raise ValueError(f"mixed devices in {devices[devices > 1].index.tolist()}")
    grouped = runs.groupby(["config", "pct"])
    out = grouped[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = grouped.size()
    out["device"] = grouped["device"].first()
    return out.reset_index()


def markdown_table(agg: pd.DataFrame, order: list[str] | None = None) -> str:
    """Accuracy in % and training time in minutes, formatted as mean ± std."""
    if order:
        agg = agg.set_index("config").loc[[c for c in order if c in set(agg["config"])]].reset_index()
    lines = ["| Config | Data | Test accuracy (%) | Macro F1 (%) | Train time (min) | Seeds |",
             "|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['config']} | {r['pct']} % "
            f"| {100 * r['test_accuracy_mean']:.2f} ± {100 * r['test_accuracy_std']:.2f} "
            f"| {100 * r['test_f1_macro_mean']:.2f} ± {100 * r['test_f1_macro_std']:.2f} "
            f"| {r['train_time_total_s_mean'] / 60:.1f} ± {r['train_time_total_s_std'] / 60:.1f} "
            f"| {r['n_seeds']} |")
    return "\n".join(lines)
