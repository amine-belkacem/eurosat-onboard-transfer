"""Dataset figures: images per class and split, and a grid of examples.

Reads splits/eurosat_rgb.csv and the extracted images, writes PNGs to figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from eot.splits import CLASSES, SPLITS

# Project palette: teal, violet, magenta first; blue/orange only as a last resort.
SPLIT_COLORS = {"train": "#1baf7a", "val": "#6250d6", "test": "#e87ba4"}
EXAMPLES_PER_CLASS = 5
EXAMPLE_SEED = 0


def plot_class_counts(df: pd.DataFrame, out: Path) -> None:
    table = (df.pivot_table(index="label", columns="split", values="path", aggfunc="count")
               .reindex(index=list(CLASSES), columns=list(SPLITS)))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    left = pd.Series(0, index=table.index)
    for split in SPLITS:
        bars = ax.barh(table.index, table[split], left=left, color=SPLIT_COLORS[split],
                       label=f"{split} ({table[split].sum()})")
        ax.bar_label(bars, labels=table[split].tolist(), label_type="center", fontsize=8,
                     color="white" if split == "val" else "black")
        left += table[split]
    ax.invert_yaxis()
    ax.set_xlabel("Images")
    ax.set_title("EuroSAT RGB: frozen stratified split (70 / 15 / 15)", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(SPLITS), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_examples(df: pd.DataFrame, root: Path, out: Path) -> None:
    # One column per class keeps the figure wide, which suits a README.
    train = df[df["split"] == "train"]
    fig, axes = plt.subplots(EXAMPLES_PER_CLASS, len(CLASSES),
                             figsize=(len(CLASSES) * 1.15, EXAMPLES_PER_CLASS * 1.15 + 0.7),
                             squeeze=False)
    for col, label in enumerate(CLASSES):
        picks = train[train["label"] == label].sample(EXAMPLES_PER_CLASS, random_state=EXAMPLE_SEED)
        for row, path in enumerate(picks["path"]):
            ax = axes[row, col]
            ax.imshow(Image.open(root / path))
            ax.set_xticks([])
            ax.set_yticks([])
        # Split CamelCase names so long labels fit above a narrow column.
        title = label.replace("Vegetation", "\nVegetation").replace("Crop", "\nCrop")
        axes[0, col].set_title(title, fontsize=8)
    fig.suptitle("EuroSAT RGB: random training examples (64 x 64 px, 10 m/px)", fontsize=10)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-csv", type=Path, default=Path("splits/eurosat_rgb.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/EuroSAT_RGB/EuroSAT_RGB"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_class_counts(df, args.out_dir / "dataset_split.png")
    plot_examples(df, args.root, args.out_dir / "dataset_examples.png")
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
