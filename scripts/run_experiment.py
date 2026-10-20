"""Run one config over several data fractions and seeds; one JSON per run.

Existing result files are skipped, so a Kaggle session that dies halfway can
simply be restarted.

    python scripts/run_experiment.py --config configs/resnet50_ft.yaml --pcts 100 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import torch

from eot.experiment import Dataset, ExperimentConfig, result_path, run, save
from eot.splits import FRACTIONS_PCT, SUBSET_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pcts", type=int, nargs="+", default=list(FRACTIONS_PCT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SUBSET_SEEDS))
    parser.add_argument("--data-root", type=Path, default=Path("data/EuroSAT_RGB/EuroSAT_RGB"))
    parser.add_argument("--split-csv", type=Path, default=Path("splits/eurosat_rgb.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny step budget and 200 eval images, to check the pipeline on CPU")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    eval_limit = None
    out_dir = args.out_dir
    if args.smoke:
        cfg = dataclasses.replace(cfg, loop=dataclasses.replace(
            cfg.loop, max_epochs=1, min_steps=6, n_evals=2, patience=2, batch_size=16, eval_batch_size=50))
        eval_limit, out_dir = 200, args.out_dir / "_smoke"

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Loading images from {args.data_root} ...", flush=True)
    ds = Dataset.load(args.split_csv, args.data_root)

    for pct in args.pcts:
        for seed in args.seeds:
            path = result_path(out_dir, cfg.name, pct, seed)
            if path.exists() and not args.overwrite:
                print(f"skip {path} (exists)")
                continue
            t0 = time.perf_counter()
            res = run(cfg, ds, pct, seed, device, eval_limit=eval_limit)
            save(res, path)
            print(f"{cfg.name} {pct:3d}% seed {seed}: val {res['best_val_acc']:.4f} "
                  f"test {res['test_accuracy']:.4f} train {res['train_time_total_s']:.0f}s "
                  f"(wall {time.perf_counter() - t0:.0f}s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
