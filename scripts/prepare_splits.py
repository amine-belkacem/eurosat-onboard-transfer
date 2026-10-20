"""Build the frozen EuroSAT split and write it to splits/eurosat_rgb.csv.

Run once after download_eurosat.py. The CSV is committed, so the split never
changes between machines (PC, Kaggle, Raspberry Pi).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eot.splits import CLASSES, FRACTIONS_PCT, SPLITS, fingerprint, list_images, make_split, select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/EuroSAT_RGB/EuroSAT_RGB"))
    parser.add_argument("--out", type=Path, default=Path("splits/eurosat_rgb.csv"))
    args = parser.parse_args()

    df = make_split(list_images(args.root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")

    print(f"{len(df)} images written to {args.out}\n")
    print(df.pivot_table(index="label", columns="split", values="path", aggfunc="count")
            .reindex(index=list(CLASSES), columns=list(SPLITS)).to_string())
    print()
    for split in SPLITS:
        print(f"{split:5s} fingerprint {fingerprint(select(df, split)['path'])}")
    # Sizes do not depend on the seed, only which images are picked does.
    print("\nTraining subset sizes:")
    for pct in FRACTIONS_PCT:
        print(f"  {pct:3d} %  {len(select(df, 'train', pct)):6d} images")


if __name__ == "__main__":
    main()
