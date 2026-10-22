"""Result figures built from results/*/pct*_seed*.json (no hand-entered numbers).

    python scripts/plot_results.py --pct 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from eot.report import aggregate, load_runs
from eot.splits import CLASSES

# Project palette, in the order configs were added. Blue/orange only as a last resort.
CONFIG_STYLE = {
    "cnn_scratch": ("CNN from scratch", "#1baf7a"),
    "resnet50_probe": ("ResNet-50 linear probe", "#6250d6"),
    "resnet50_ft": ("ResNet-50 fine-tuned", "#e87ba4"),
    "vit_small_probe": ("ViT-S/16 linear probe", "#eda100"),
    "vit_small_ft": ("ViT-S/16 fine-tuned", "#008300"),
}
WHITE_TO_TEAL = LinearSegmentedColormap.from_list("white_teal", ["#ffffff", "#1baf7a", "#0b4f37"])


def plot_accuracy_and_time(results_dir: Path, pct: int, out: Path) -> None:
    runs = load_runs(results_dir)
    runs = runs[runs["pct"] == pct]
    agg = aggregate(runs).set_index("config")
    configs = [c for c in CONFIG_STYLE if c in agg.index]
    labels = [CONFIG_STYLE[c][0] for c in configs]
    colors = [CONFIG_STYLE[c][1] for c in configs]
    y = np.arange(len(configs))

    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(11, 0.8 * len(configs) + 1.6), sharey=True)

    # Accuracy: the axis does not start at 0, so dots (one per seed) and mean ± std, no bars.
    for i, c in enumerate(configs):
        seeds = 100 * runs.loc[runs["config"] == c, "test_accuracy"].to_numpy()
        mean, std = 100 * agg.loc[c, "test_accuracy_mean"], 100 * agg.loc[c, "test_accuracy_std"]
        ax_acc.errorbar(mean, i, xerr=std, fmt="D", color=colors[i], markersize=8, capsize=5, zorder=3)
        ax_acc.scatter(seeds, np.full(len(seeds), i + 0.18), s=18, color=colors[i], alpha=0.7,
                       edgecolor="black", linewidth=0.4, zorder=2)
        ax_acc.annotate(f"{mean:.2f} ± {std:.2f}", (mean, i - 0.28), ha="center", fontsize=9)
    ax_acc.set_yticks(y, labels)
    ax_acc.invert_yaxis()
    ax_acc.set_xlabel("Test accuracy (%)  -  diamond: mean ± std, dots: individual seeds")
    ax_acc.grid(axis="x", alpha=0.3)

    # Training time on a T4, bars from zero.
    means = agg.loc[configs, "train_time_total_s_mean"].to_numpy() / 60
    stds = agg.loc[configs, "train_time_total_s_std"].to_numpy() / 60
    bars = ax_time.barh(y, means, xerr=stds, color=colors, capsize=4)
    ax_time.bar_label(bars, labels=[f"{m:.1f} min" for m in means], padding=6, fontsize=9)
    ax_time.set_xlabel(f"Training time per run (min), {agg['device'].iloc[0]}")
    ax_time.set_xlim(0, means.max() * 1.25)
    ax_time.grid(axis="x", alpha=0.3)

    for ax in (ax_acc, ax_time):
        ax.spines[["top", "right"]].set_visible(False)
    n_seeds = int(agg["n_seeds"].min())
    fig.suptitle(f"EuroSAT RGB, {pct} % of the training set, test set of 4,050 images, {n_seeds} seeds")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summed_confusion(results_dir: Path, config: str, pct: int) -> np.ndarray:
    files = sorted((results_dir / config).glob(f"pct{pct:03d}_seed*.json"))
    if not files:
        raise FileNotFoundError(f"no runs for {config} at {pct} %")
    return sum(np.array(json.loads(f.read_text())["test_confusion_matrix"]) for f in files)


def plot_confusions(results_dir: Path, configs: list[str], pct: int, out: Path) -> None:
    fig, axes = plt.subplots(1, len(configs), figsize=(6.2 * len(configs), 6))
    for ax, config in zip(np.atleast_1d(axes), configs):
        cm = summed_confusion(results_dir, config, pct)
        pct_rows = 100 * cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct_rows, cmap=WHITE_TO_TEAL, vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j] and (i == j or cm[i, j] >= 5):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if pct_rows[i, j] > 60 else "black")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(CONFIG_STYLE[config][0], fontsize=10)
    fig.suptitle(f"Test confusions summed over seeds ({pct} % of the data). "
                 "Counts shown on the diagonal and when 5 or more", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--pct", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_and_time(args.results_dir, args.pct, args.out_dir / f"accuracy_time_pct{args.pct:03d}.png")
    plot_confusions(args.results_dir, ["resnet50_probe", "resnet50_ft"], args.pct,
                    args.out_dir / f"confusion_resnet50_pct{args.pct:03d}.png")
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
