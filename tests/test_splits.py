"""Tests for the frozen split and the nested low-data subsets.

Expected values are worked out by hand in the comments. Two groups of tests:
synthetic folders (no dataset needed) and checks on the committed CSV, which
is in the repository and therefore available in CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eot.splits import (
    CLASSES,
    FRACTIONS_PCT,
    SUBSET_SEEDS,
    fingerprint,
    list_images,
    make_split,
    select,
    split_sizes,
    subset_size,
)

FROZEN_CSV = Path(__file__).resolve().parents[1] / "splits" / "eurosat_rgb.csv"


def check(got, expected, what=""):
    assert got == expected, f"ECHEC {what}: obtenu {got}, attendu {expected}"


def make_tree(root: Path, n_per_class: int) -> Path:
    """Create empty image files named like EuroSAT (Label_1.jpg ... Label_n.jpg)."""
    for label in CLASSES:
        (root / label).mkdir(parents=True)
        for i in range(1, n_per_class + 1):
            (root / label / f"{label}_{i}.jpg").touch()
    return root


@pytest.fixture(scope="module")
def frozen() -> pd.DataFrame:
    return pd.read_csv(FROZEN_CSV)


# ---------------------------------------------------------------- hand-computed sizes

@pytest.mark.parametrize("n, expected", [
    (3000, (2100, 450, 450)),   # 3000*15//100 = 450 ; 3000-900 = 2100
    (2500, (1750, 375, 375)),   # 2500*15//100 = 375 ; 2500-750 = 1750
    (2000, (1400, 300, 300)),   # 2000*15//100 = 300 ; 2000-600 = 1400
    (101, (71, 15, 15)),        # 101*15 = 1515, //100 = 15 ; 101-30 = 71
])
def test_split_sizes(n, expected):
    check(split_sizes(n), expected, f"split_sizes({n})")


@pytest.mark.parametrize("n_train, pct, expected", [
    (2100, 1, 21),      # (2100*1 + 50)//100 = 2150//100 = 21
    (1750, 1, 18),      # 17.5 rounds half up: (1750+50)//100 = 18
    (1400, 1, 14),      # (1400+50)//100 = 14
    (1750, 5, 88),      # 87.5 -> (8750+50)//100 = 88
    (1750, 25, 438),    # 437.5 -> (43750+50)//100 = 438
    (2100, 100, 2100),
    (10, 1, 1),         # 0.1 rounds to 0, floor of one image per class
])
def test_subset_size(n_train, pct, expected):
    check(subset_size(n_train, pct), expected, f"subset_size({n_train}, {pct})")


@pytest.mark.parametrize("pct", [0, -5, 101])
def test_subset_size_rejects_bad_pct(pct):
    with pytest.raises(ValueError):
        subset_size(100, pct)


# ---------------------------------------------------------------- synthetic folders

def test_list_images_natural_order(tmp_path):
    df = list_images(make_tree(tmp_path, 12))
    first = df[df["label"] == "AnnualCrop"]["path"].tolist()[:3]
    check(first, ["AnnualCrop/AnnualCrop_1.jpg", "AnnualCrop/AnnualCrop_2.jpg",
                  "AnnualCrop/AnnualCrop_3.jpg"], "ordre des fichiers")
    check(df[df["label"] == "AnnualCrop"]["path"].iloc[-1],
          "AnnualCrop/AnnualCrop_12.jpg", "dernier fichier (12 apres 2)")


def test_list_images_missing_class(tmp_path):
    make_tree(tmp_path, 3)
    for f in (tmp_path / "River").iterdir():
        f.unlink()
    (tmp_path / "River").rmdir()
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path)


def test_make_split_deterministic(tmp_path):
    images = list_images(make_tree(tmp_path, 40))
    a, b = make_split(images, seed=0), make_split(images, seed=0)
    check(a.equals(b), True, "meme graine => meme decoupage")
    c = make_split(images, seed=1)
    check(a["split"].equals(c["split"]), False, "graine differente => decoupage different")


def test_ranks_are_a_permutation_of_train_rows(tmp_path):
    # 40 images per class -> 40 - 6 - 6 = 28 training images per class.
    df = make_split(list_images(make_tree(tmp_path, 40)))
    for s in SUBSET_SEEDS:
        for label in CLASSES:
            train = df[(df["label"] == label) & (df["split"] == "train")]
            check(sorted(train[f"rank_s{s}"]), list(range(28)), f"rangs {label} graine {s}")
        check(set(df[df["split"] != "train"][f"rank_s{s}"]), {-1}, "rang hors train")


# ---------------------------------------------------------------- committed CSV

def test_frozen_split_totals(frozen):
    # train: 5*2100 + 4*1750 + 1400 = 10500 + 7000 + 1400 = 18900
    # val = test: 5*450 + 4*375 + 300 = 2250 + 1500 + 300 = 4050
    counts = frozen["split"].value_counts().to_dict()
    check(counts, {"train": 18900, "val": 4050, "test": 4050}, "effectifs train/val/test")


def test_frozen_split_disjoint(frozen):
    check(frozen["path"].is_unique, True, "chaque image apparait une seule fois")
    check(len(frozen), 27000, "nombre total d'images")


def test_frozen_split_stratified(frozen):
    table = frozen.pivot_table(index="label", columns="split", values="path", aggfunc="count")
    for label in CLASSES:
        n = int(table.loc[label].sum())
        check(tuple(int(table.loc[label, s]) for s in ("train", "val", "test")),
              split_sizes(n), f"stratification {label}")


# 5 classes of 2100, 4 of 1750 and 1 of 1400 training images:
#   1 %   5*21  + 4*18  + 14   = 105 + 72 + 14     = 191
#   5 %   5*105 + 4*88  + 70   = 525 + 352 + 70    = 947
#   10 %  5*210 + 4*175 + 140  = 1050 + 700 + 140  = 1890
#   25 %  5*525 + 4*438 + 350  = 2625 + 1752 + 350 = 4727
#   100 %                                         = 18900
EXPECTED_SUBSET_SIZES = {1: 191, 5: 947, 10: 1890, 25: 4727, 100: 18900}


@pytest.mark.parametrize("seed", SUBSET_SEEDS)
def test_subset_sizes(frozen, seed):
    for pct in FRACTIONS_PCT:
        check(len(select(frozen, "train", pct, seed)), EXPECTED_SUBSET_SIZES[pct],
              f"taille du sous-ensemble {pct} % graine {seed}")


@pytest.mark.parametrize("seed", SUBSET_SEEDS)
def test_subsets_are_nested(frozen, seed):
    previous: set[str] = set()
    for pct in FRACTIONS_PCT:
        current = set(select(frozen, "train", pct, seed)["path"])
        check(previous <= current, True, f"{pct} % contient la fraction precedente")
        previous = current


def test_subsets_differ_between_seeds(frozen):
    a = set(select(frozen, "train", 1, 0)["path"])
    b = set(select(frozen, "train", 1, 1)["path"])
    check(a != b, True, "les graines tirent des images differentes a 1 %")


def test_frozen_fingerprints(frozen):
    # Pinned when the split was created. If this fails, the split changed and
    # every published number is no longer comparable.
    check(fingerprint(select(frozen, "train")["path"]), "d3c429963249f11c", "empreinte train")
    check(fingerprint(select(frozen, "val")["path"]), "83851715fdc47678", "empreinte val")
    check(fingerprint(select(frozen, "test")["path"]), "4072469443577629", "empreinte test")
