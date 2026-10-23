"""ONNX export, INT8 quantization and benchmark helpers (CPU, synthetic data, no download)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from eot import bench, data, deploy, models, training
from eot.experiment import Dataset, ExperimentConfig, run, save_checkpoint
from eot.splits import CLASSES, list_images, make_split

ROOT = Path(__file__).resolve().parents[1]


def check(got, expected, what=""):
    assert got == expected, f"ECHEC {what}: obtenu {got}, attendu {expected}"


def check_close(got, expected, what="", tol=1e-9):
    assert abs(got - expected) <= tol, f"ECHEC {what}: obtenu {got}, attendu {expected}"


# ---------------------------------------------------------------- benchmark statistics

def test_latency_stats_hand_computed():
    # latencies 1, 2, ..., 20 ms
    s = bench.latency_stats(np.arange(1, 21))
    check_close(s["mean_ms"], 10.5, "moyenne")
    check_close(s["median_ms"], 10.5, "mediane (10 + 11) / 2")
    # p95: position 0.95 * 19 = 18.05 in the sorted list (0-based) -> 19 + 0.05 * (20 - 19) = 19.05
    check_close(s["p95_ms"], 19.05, "p95")
    # p99: position 0.99 * 19 = 18.81 -> 19 + 0.81 = 19.81
    check_close(s["p99_ms"], 19.81, "p99")
    # sum of squared deviations for 1..n is n(n^2-1)/12 = 20*399/12 = 665 ; std = sqrt(665/19) = sqrt(35)
    check_close(s["std_ms"], math.sqrt(35), "ecart-type", 1e-12)


def test_parse_and_decode_throttled():
    value = bench.parse_throttled("throttled=0x50005\n")
    check(value, 0x50005, "valeur hexadecimale")
    # 0x50005 = bits 0, 2, 16 and 18
    flags = bench.decode_throttled(value)
    on = sorted(k for k, v in flags.items() if v)
    check(on, ["throttled_now", "throttled_occurred", "under_voltage_now", "under_voltage_occurred"],
          "drapeaux actifs")


def test_parse_throttled_rejects_garbage():
    with pytest.raises(ValueError):
        bench.parse_throttled("error=1")


@pytest.mark.parametrize("freqs, before, after, valid", [
    ([2400.0, 2400.0, 2400.0], 0, 0, True),            # steady clock, no event
    ([2400.0, 1500.0, 2400.0], 0, 0, False),           # clock dropped during the run
    ([2400.0, 2400.0], 0, 0x40000, False),             # throttling occurred during the run (bit 18)
    ([2400.0, 2400.0], 0x50000, 0x50000, True),        # old events from before the run do not count
    ([None, None], None, None, False),                 # PC dry run: nothing could be checked
])
def test_run_is_valid(freqs, before, after, valid):
    got, reasons = bench.run_is_valid(freqs, before, after)
    check(got, valid, f"validite {freqs} {before} {after} ({reasons})")


# ---------------------------------------------------------------- export

@pytest.mark.parametrize("input_size", [64, 112])
def test_onnx_matches_pytorch(tmp_path, input_size):
    torch.manual_seed(0)
    module = deploy.TileClassifier(models.SmallCNN(), input_size).eval()
    path = tmp_path / "m.onnx"
    deploy.export_onnx(module, path)
    tiles = np.random.default_rng(0).integers(0, 256, (8, 3, 64, 64), dtype=np.uint8)
    eq = deploy.check_equivalence(module, path, tiles)
    assert eq["max_abs_diff"] < 1e-4, f"ECHEC ecart PyTorch/ONNX: obtenu {eq['max_abs_diff']}, attendu < 1e-4"
    check(eq["argmax_agreement"], 1.0, "memes classes predites")
    # SmallCNN has 1,175,786 parameters; folding BatchNorm into the convolutions
    # changes that by well under 3 %.
    ratio = deploy.onnx_parameter_count(path) / models.count_parameters(module.model)
    assert ratio > 0.97, f"ECHEC parametres conserves: obtenu {ratio:.3f}, attendu > 0.97"


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = models.SmallCNN().eval()
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "cnn_scratch.yaml")
    save_checkpoint(tmp_path / "c.pt", cfg, model.state_dict(), test_accuracy=0.5)
    module, meta = deploy.load_checkpoint(tmp_path / "c.pt")
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        same = torch.equal(module(x), model(data.preprocess(x, 64)))
    check(same, True, "sorties identiques apres rechargement")
    check((meta["config"], meta["input_size"]), ("cnn_scratch", 64), "metadonnees")


def colour_tiles(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Tiles whose class is encoded by a mean colour plus noise: easy to learn, real decision boundaries."""
    rng = np.random.default_rng(seed)
    centres = np.linspace(20, 235, len(CLASSES))
    tiles, labels = [], []
    for c, v in enumerate(centres):
        base = np.array([v, 255 - v, (v * 3) % 256])[:, None, None]
        noise = rng.normal(0, 12, (n_per_class, 3, 64, 64))
        tiles.append(np.clip(base + noise, 0, 255).astype(np.uint8))
        labels += [c] * n_per_class
    return np.concatenate(tiles), np.array(labels)


def test_int8_agrees_with_fp32(tmp_path):
    torch.manual_seed(0)
    x_tr, y_tr = colour_tiles(8, seed=0)
    model = models.SmallCNN()
    cfg = training.LoopConfig(batch_size=40, lr_head=3e-3, lr_backbone=None, weight_decay=0.0,
                              max_epochs=30, min_steps=0, n_evals=5, patience=5, amp=False)
    x_t, y_t = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    prep = lambda xb, train: data.preprocess(xb, 64)
    res = training.fit(model, models.head_parameters(model), x_t, y_t, x_t, y_t, prep, cfg,
                       torch.device("cpu"), seed=0)
    model.load_state_dict(res["best_state"])
    module = deploy.TileClassifier(model, 64).eval()

    fp32, int8 = tmp_path / "m_fp32.onnx", tmp_path / "m_int8.onnx"
    deploy.export_onnx(module, fp32)
    calib, _ = colour_tiles(4, seed=1)
    deploy.quantize_int8(fp32, int8, calib, batch_size=8)

    import onnxruntime as ort
    x_new, y_new = colour_tiles(10, seed=2)
    run_model = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(
        None, {deploy.INPUT_NAME: x_new})[0].argmax(1)
    p32, p8 = run_model(fp32), run_model(int8)
    agreement = float((p32 == p8).mean())
    assert agreement >= 0.95, f"ECHEC accord FP32/INT8: obtenu {agreement:.2f}, attendu >= 0.95"
    assert int8.stat().st_size < fp32.stat().st_size / 2, \
        f"ECHEC taille INT8: obtenu {int8.stat().st_size} octets, attendu < {fp32.stat().st_size // 2}"


def test_probe_checkpoint_refused(tmp_path):
    df = make_split(list_images(_tiny_tree(tmp_path / "d")))
    ds = Dataset(df, torch.zeros(len(df), 3, 64, 64, dtype=torch.uint8), data.labels_of(df))
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "resnet50_probe.yaml")
    with pytest.raises(ValueError):
        run(cfg, ds, 100, 0, torch.device("cpu"), pretrained=False, checkpoint_path=tmp_path / "p.pt")


# ---------------------------------------------------------------- benchmark end to end

def _tiny_tree(root: Path, n: int = 20) -> Path:
    rng = np.random.default_rng(0)
    for label in CLASSES:
        (root / label).mkdir(parents=True)
        for i in range(1, n + 1):
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / label / f"{label}_{i}.jpg")
    return root


def test_bench_one_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location("bench_onnx", ROOT / "scripts" / "bench_onnx.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    root = _tiny_tree(tmp_path / "data")
    split_csv = tmp_path / "split.csv"
    make_split(list_images(root)).to_csv(split_csv, index=False)
    torch.manual_seed(0)
    onnx_path = tmp_path / "cnn_fp32.onnx"
    deploy.export_onnx(deploy.TileClassifier(models.SmallCNN(), 64).eval(), onnx_path)

    res = script.bench_one(str(onnx_path), str(split_csv), str(root), threads=1, warmup=2, runs=10)
    check(res["latency"]["n"], 10, "nombre de mesures de latence")
    check(len(res["latencies_ms"]), 10, "latences enregistrees")
    # 20 images per class -> 3 test images per class -> 30
    check(res["n_test"], 30, "taille du test")
    check(int(np.sum(res["test_confusion_matrix"])), 30, "somme de la matrice de confusion")
