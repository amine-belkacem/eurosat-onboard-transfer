"""Frozen train/val/test split and nested low-data subsets for EuroSAT.

Two design decisions drive this module:

* The train/val/test split is drawn once with a fixed seed and saved to a CSV
  that lives in the repository. Every experiment reads that file, so no run can
  quietly see a different test set.
* Low-data subsets (1 %, 5 %, ...) are nested: the 1 % subset is the first rows
  of the 5 % subset, and so on. Each class gets its own shuffled ranking of its
  training images, one ranking per training seed. Taking the first k ranked
  images of every class gives a stratified subset, and growing k can only add
  images. Using one ranking per seed means the reported spread across seeds
  also covers "which images did I happen to label", which matters a lot at 1 %.

All sizes are computed with integer arithmetic so they can be checked by hand.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
SPLITS = ("train", "val", "test")
VAL_PCT = 15
TEST_PCT = 15
FRACTIONS_PCT = (1, 5, 10, 25, 100)
SPLIT_SEED = 0
SUBSET_SEEDS = (0, 1, 2)
IMAGE_SUFFIXES = (".jpg", ".tif")

_INDEX_RE = re.compile(r"_(\d+)\.\w+$")


def _image_index(path: str) -> int:
    match = _INDEX_RE.search(path)
    if match is None:
        raise ValueError(f"cannot read image index from {path!r}")
    return int(match.group(1))


def list_images(root: Path) -> pd.DataFrame:
    """Return one row per image with its path relative to `root` and its label.

    Rows are sorted by class, then by the numeric index in the file name
    (AnnualCrop_2 before AnnualCrop_10), so the order does not depend on the
    file system.
    """
    rows = []
    for label in CLASSES:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class folder: {class_dir}")
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        if not files:
            raise FileNotFoundError(f"no images in {class_dir}")
        for p in sorted(files, key=lambda p: _image_index(p.name)):
            rows.append((f"{label}/{p.name}", label))
    return pd.DataFrame(rows, columns=["path", "label"])


def split_sizes(n: int) -> tuple[int, int, int]:
    """Number of (train, val, test) images for a class of `n` images.

    Val and test are rounded down, train takes the remainder.
    """
    n_val = n * VAL_PCT // 100
    n_test = n * TEST_PCT // 100
    return n - n_val - n_test, n_val, n_test


def subset_size(n_train: int, pct: int) -> int:
    """Images kept from a class with `n_train` training images at `pct` percent.

    Rounded half up, with at least one image so no class disappears.
    """
    if not 0 < pct <= 100:
        raise ValueError(f"pct must be in (0, 100], got {pct}")
    return max(1, (n_train * pct + 50) // 100)


def make_split(images: pd.DataFrame, seed: int = SPLIT_SEED,
               subset_seeds: tuple[int, ...] = SUBSET_SEEDS) -> pd.DataFrame:
    """Assign each image to train/val/test and rank training images per seed.

    Adds a `split` column and one `rank_s{seed}` column per subset seed. Ranks
    are 0..n_train-1 within each class for training rows and -1 elsewhere.
    """
    rng = np.random.default_rng(seed)
    out = images.copy()
    out["split"] = ""
    for label in CLASSES:
        idx = out.index[out["label"] == label].to_numpy()
        n_train, n_val, _ = split_sizes(len(idx))
        perm = rng.permutation(idx)
        out.loc[perm[:n_train], "split"] = "train"
        out.loc[perm[n_train:n_train + n_val], "split"] = "val"
        out.loc[perm[n_train + n_val:], "split"] = "test"

    for s in subset_seeds:
        col = f"rank_s{s}"
        out[col] = -1
        sub_rng = np.random.default_rng([seed, s])
        for label in CLASSES:
            idx = out.index[(out["label"] == label) & (out["split"] == "train")].to_numpy()
            out.loc[idx, col] = sub_rng.permutation(len(idx))
    return out


def select(split_df: pd.DataFrame, split: str, pct: int = 100,
           seed: int = SUBSET_SEEDS[0]) -> pd.DataFrame:
    """Rows of `split`. For train, keep the first `pct` percent of each class."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    rows = split_df[split_df["split"] == split]
    if split != "train" or pct == 100:
        return rows.reset_index(drop=True)
    col = f"rank_s{seed}"
    if col not in rows:
        raise KeyError(f"no ranking for seed {seed}; available: "
                       f"{[c for c in rows if c.startswith('rank_s')]}")
    parts = []
    for label in CLASSES:
        cls = rows[rows["label"] == label]
        parts.append(cls[cls[col] < subset_size(len(cls), pct)])
    return pd.concat(parts).reset_index(drop=True)


def fingerprint(paths: pd.Series) -> str:
    """Short SHA-256 of a set of paths, independent of row order."""
    joined = "\n".join(sorted(paths)).encode()
    return hashlib.sha256(joined).hexdigest()[:16]
