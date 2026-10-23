"""Export trained checkpoints to ONNX (FP32 and INT8) and check them.

For every checkpoint:
1. export the TileClassifier (preprocessing included) to FP32 ONNX;
2. check that ONNX holds as many parameters as PyTorch (catches constant
   folding of whole branches) and that ONNX Runtime matches PyTorch on
   validation tiles;
3. quantize to INT8, calibrated on validation tiles only (never test).

Writes deploy/onnx/<config>_{fp32,int8}.onnx and a manifest with sizes,
checks and SHA-256 sums, so the files copied to the Raspberry Pi can be
verified there.

    python scripts/export_models.py --checkpoints deploy/checkpoints/*.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eot import data, deploy, models
from eot.splits import CLASSES

CALIBRATION_PER_CLASS = 50   # 500 validation tiles
EQUIVALENCE_TILES = 64
CALIBRATION_SEED = 0
MIN_PARAM_RATIO = 0.97       # BatchNorm folding changes the count by well under 1 %
MAX_ABS_DIFF = 1e-3          # logits, FP32 PyTorch vs ONNX Runtime


def validation_tiles(split_csv: Path, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Stratified calibration tiles and a separate set of equivalence tiles, both from val."""
    df = pd.read_csv(split_csv)
    val = df[df["split"] == "val"]
    calib = pd.concat([val[val["label"] == c].sample(CALIBRATION_PER_CLASS, random_state=CALIBRATION_SEED)
                       for c in CLASSES])
    rest = val.drop(calib.index).sample(EQUIVALENCE_TILES, random_state=CALIBRATION_SEED)
    load = lambda rows: data.load_images(root, rows["path"]).numpy()
    return load(calib), load(rest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("deploy/onnx"))
    parser.add_argument("--data-root", type=Path, default=Path("data/EuroSAT_RGB/EuroSAT_RGB"))
    parser.add_argument("--split-csv", type=Path, default=Path("splits/eurosat_rgb.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("results/onnx_manifest.json"))
    args = parser.parse_args()

    calib_tiles, check_tiles = validation_tiles(args.split_csv, args.data_root)
    manifest, failures = {}, []
    for ckpt_path in sorted(args.checkpoints):
        module, meta = deploy.load_checkpoint(ckpt_path)
        name = meta["config"]
        fp32 = args.out_dir / f"{name}_fp32.onnx"
        int8 = args.out_dir / f"{name}_int8.onnx"
        print(f"{name}: export ...", flush=True)
        deploy.export_onnx(module, fp32)

        torch_params = models.count_parameters(module.model)
        onnx_params = deploy.onnx_parameter_count(fp32)
        ratio = onnx_params / torch_params
        equiv = deploy.check_equivalence(module, fp32, check_tiles)
        if ratio < MIN_PARAM_RATIO:
            failures.append(f"{name}: ONNX keeps {onnx_params} of {torch_params} parameters ({ratio:.3f})")
        if equiv["max_abs_diff"] > MAX_ABS_DIFF or equiv["argmax_agreement"] < 1.0:
            failures.append(f"{name}: ONNX differs from PyTorch {equiv}")

        print(f"{name}: INT8 calibration on {len(calib_tiles)} validation tiles ...", flush=True)
        deploy.quantize_int8(fp32, int8, calib_tiles)

        manifest[name] = {
            "checkpoint": ckpt_path.name,
            "checkpoint_test_accuracy": meta["test_accuracy"],
            "trained_at_commit": meta["git_commit"],
            "input_size": meta["input_size"],
            "torch_params": torch_params,
            "onnx_params": onnx_params,
            "equivalence": equiv,
            "calibration_tiles": int(len(calib_tiles)),
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": deploy.sha256(p)} for p in (fp32, int8)},
        }
        print(f"{name}: params {onnx_params}/{torch_params}, max diff {equiv['max_abs_diff']:.2e}, "
              f"FP32 {fp32.stat().st_size / 1e6:.1f} MB, INT8 {int8.stat().st_size / 1e6:.1f} MB", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Manifest written to {args.manifest}")
    for f in failures:
        print(f"ECHEC {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
