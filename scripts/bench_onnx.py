"""Measure ONNX models on the Raspberry Pi 5: latency, memory and test accuracy.

Each model runs in its own fresh process, so the peak memory reported belongs
to that model alone. Per model:

* latency: batch 1, real test tiles, `--warmup` untimed runs then `--runs`
  timed runs; the CPU temperature and clock are sampled every 25 runs, and
  `vcgencmd get_throttled` is read before and after. The run is flagged
  invalid if the clock moved or any throttling happened;
* accuracy on the full test set (4,050 tiles), batch 64;
* model file size and peak resident memory.

    python scripts/bench_onnx.py --models deploy/onnx/*.onnx --out-dir results/pi
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eot import bench
from eot.metrics import summarize
from eot.splits import CLASSES

SAMPLE_EVERY = 25
ACCURACY_BATCH = 64
LATENCY_TILES = 500   # distinct test tiles cycled through during the latency loop


def load_test_tiles(split_csv: Path, root: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(split_csv)
    test = df[df["split"] == "test"].reset_index(drop=True)
    tiles = np.stack([np.asarray(Image.open(root / p).convert("RGB")) for p in test["path"]])
    labels = test["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return np.ascontiguousarray(tiles.transpose(0, 3, 1, 2)), labels


def rss_mb() -> tuple[float | None, float | None]:
    """(current, peak) resident memory of this process in MB, Linux only."""
    try:
        status = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return None, None
    fields = {line.split(":")[0]: line.split()[1] for line in status if line.startswith(("VmRSS", "VmHWM"))}
    return int(fields["VmRSS"]) / 1024, int(fields["VmHWM"]) / 1024


def bench_one(model_path: str, split_csv: str, data_root: str, threads: int, warmup: int, runs: int) -> dict:
    import onnxruntime as ort

    tiles, labels = load_test_tiles(Path(split_csv), Path(data_root))
    rss_before, _ = rss_mb()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    name_in = sess.get_inputs()[0].name

    pick = np.linspace(0, len(tiles) - 1, LATENCY_TILES).astype(int)
    lat_tiles = [tiles[i:i + 1] for i in pick]
    for i in range(warmup):
        sess.run(None, {name_in: lat_tiles[i % len(lat_tiles)]})

    throttled_before = bench.read_throttled()
    temps, freqs, latencies = [], [], []
    for i in range(runs):
        if i % SAMPLE_EVERY == 0:
            temps.append(bench.read_temperature_c())
            freqs.append(bench.read_cpu_freq_mhz())
        x = lat_tiles[i % len(lat_tiles)]
        t0 = time.perf_counter_ns()
        sess.run(None, {name_in: x})
        latencies.append((time.perf_counter_ns() - t0) / 1e6)
    temps.append(bench.read_temperature_c())
    freqs.append(bench.read_cpu_freq_mhz())
    throttled_after = bench.read_throttled()
    valid, reasons = bench.run_is_valid(freqs, throttled_before, throttled_after)

    preds = np.concatenate([sess.run(None, {name_in: tiles[i:i + ACCURACY_BATCH]})[0].argmax(1)
                            for i in range(0, len(tiles), ACCURACY_BATCH)])
    test = summarize(labels, preds, len(CLASSES))
    _, rss_peak = rss_mb()

    return {
        "model_file": Path(model_path).name,
        "model_bytes": Path(model_path).stat().st_size,
        "session_load_s": load_s,
        "latency": bench.latency_stats(latencies),
        "latencies_ms": [round(v, 4) for v in latencies],
        "warmup_runs": warmup,
        "temperature_c": temps,
        "cpu_freq_mhz": freqs,
        "throttled_before": throttled_before,
        "throttled_after": throttled_after,
        "valid": valid,
        "invalid_reasons": reasons,
        "rss_before_session_mb": rss_before,
        "rss_peak_mb": rss_peak,
        "test_accuracy": test["accuracy"],
        "test_f1_macro": test["f1_macro"],
        "test_confusion_matrix": test["confusion_matrix"],
        "n_test": int(len(labels)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/pi"))
    parser.add_argument("--data-root", type=Path, default=Path("data/EuroSAT_RGB/EuroSAT_RGB"))
    parser.add_argument("--split-csv", type=Path, default=Path("splits/eurosat_rgb.csv"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--cooldown-s", type=float, default=30.0,
                        help="pause between models so each starts from a similar temperature")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    info = bench.platform_info(args.threads)
    print(json.dumps(info, indent=1))
    ctx = mp.get_context("spawn")
    for i, model in enumerate(sorted(args.models)):
        if i:
            time.sleep(args.cooldown_s)
        print(f"{model.name}: running ...", flush=True)
        with ctx.Pool(1) as pool:
            res = pool.apply(bench_one, (str(model), str(args.split_csv), str(args.data_root),
                                         args.threads, args.warmup, args.runs))
        res.update(platform=info, measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        out = args.out_dir / f"{model.stem}.json"
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        lat = res["latency"]
        flag = "" if res["valid"] else f"  INVALID: {'; '.join(res['invalid_reasons'])}"
        print(f"{model.name}: median {lat['median_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms, "
              f"acc {100 * res['test_accuracy']:.2f} %, peak RSS {res['rss_peak_mb']} MB{flag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
