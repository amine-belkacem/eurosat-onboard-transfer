"""Collect results/*/pct*_seed*.json into results/runs.csv, results/summary.csv and a Markdown table."""

from __future__ import annotations

import argparse
from pathlib import Path

from eot.report import aggregate, load_runs, markdown_table

CONFIG_ORDER = ["cnn_scratch", "resnet50_probe", "resnet50_ft", "vit_small_probe", "vit_small_ft"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    agg = aggregate(runs)
    runs.to_csv(args.results_dir / "runs.csv", index=False, lineterminator="\n")
    agg.to_csv(args.results_dir / "summary.csv", index=False, lineterminator="\n")
    table = markdown_table(agg, CONFIG_ORDER)
    (args.results_dir / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
