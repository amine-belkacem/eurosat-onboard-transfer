"""One experiment = one (config, data fraction, seed) triple, written to one JSON file."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from eot import data, metrics, models, training
from eot.splits import CLASSES, fingerprint, select

MODES = ("scratch", "finetune", "probe")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    mode: str
    input_size: int
    loop: training.LoopConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        loop = training.LoopConfig(**raw.pop("loop"))
        cfg = cls(loop=loop, **raw)
        if cfg.mode not in MODES:
            raise ValueError(f"{path}: mode must be one of {MODES}, got {cfg.mode!r}")
        if cfg.mode == "scratch" and cfg.loop.lr_backbone is not None:
            raise ValueError(f"{path}: scratch training uses a single learning rate (lr_backbone: null)")
        return cfg


@dataclass
class Dataset:
    """All images of one split CSV, loaded once and indexed by subset."""
    split_df: pd.DataFrame
    images: torch.Tensor  # uint8 (N, 3, 64, 64), same row order as split_df
    labels: torch.Tensor

    @classmethod
    def load(cls, split_csv: Path, root: Path) -> "Dataset":
        df = pd.read_csv(split_csv)
        return cls(df, data.load_images(root, df["path"]), data.labels_of(df))

    def rows(self, split: str, pct: int = 100, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, pd.Series]:
        paths = select(self.split_df, split, pct, seed)["path"]
        row_of = pd.Series(self.split_df.index, index=self.split_df["path"])
        idx = torch.tensor(row_of.loc[paths].to_numpy())
        return self.images[idx], self.labels[idx], paths


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


def run(cfg: ExperimentConfig, ds: Dataset, pct: int, seed: int, device: torch.device,
        pretrained: bool = True, eval_limit: int | None = None) -> dict:
    """Train on `pct` % of the training split, select on val, then evaluate on test once."""
    torch.manual_seed(seed)
    x_tr, y_tr, tr_paths = ds.rows("train", pct, seed)
    x_val, y_val, _ = ds.rows("val")
    x_te, y_te, _ = ds.rows("test")
    if eval_limit is not None:
        # Smoke runs only. Rows are sorted by class, so take evenly spaced rows
        # rather than the first ones (which would all be AnnualCrop).
        pick_val = torch.linspace(0, len(y_val) - 1, eval_limit).long()
        pick_te = torch.linspace(0, len(y_te) - 1, eval_limit).long()
        x_val, y_val, x_te, y_te = x_val[pick_val], y_val[pick_val], x_te[pick_te], y_te[pick_te]

    def prep_images(xb: torch.Tensor, train: bool) -> torch.Tensor:
        if train:
            xb = data.augment(xb)
        return data.preprocess(xb, cfg.input_size)

    feature_time_s = 0.0
    if cfg.mode == "probe":
        backbone = models.create_backbone(cfg.model, pretrained).to(device)
        n_params = models.count_parameters(backbone)
        training._sync(device)
        t0 = time.perf_counter()
        f_tr = training.extract_features(backbone, x_tr, prep_images, device, cfg.loop.eval_batch_size, cfg.loop.amp)
        f_val = training.extract_features(backbone, x_val, prep_images, device, cfg.loop.eval_batch_size, cfg.loop.amp)
        training._sync(device)
        feature_time_s = time.perf_counter() - t0
        model, head_params = training.linear_probe_head(f_tr, len(CLASSES))
        n_trainable = sum(p.numel() for p in head_params)
        result = training.fit(model, head_params, f_tr, y_tr, f_val, y_val,
                              lambda fb, train: fb, cfg.loop, device, seed)
        model.load_state_dict(result["best_state"])
        f_te = training.extract_features(backbone, x_te, prep_images, device, cfg.loop.eval_batch_size, cfg.loop.amp)
        y_pred = training.predict(model, f_te, lambda fb, train: fb, device, cfg.loop.eval_batch_size, cfg.loop.amp)
    else:
        model = models.create_model(cfg.model, pretrained=pretrained and cfg.mode == "finetune")
        n_params = n_trainable = models.count_parameters(model)
        result = training.fit(model, models.head_parameters(model), x_tr, y_tr, x_val, y_val,
                              prep_images, cfg.loop, device, seed)
        model.load_state_dict(result["best_state"])
        y_pred = training.predict(model, x_te, prep_images, device, cfg.loop.eval_batch_size, cfg.loop.amp)

    test = metrics.summarize(y_te.numpy(), y_pred, len(CLASSES))
    result.pop("best_state")
    return {
        "config": cfg.name,
        "model": cfg.model,
        "timm_weights": models.TIMM_MODELS.get(cfg.model),
        "pretrained": pretrained and cfg.mode != "scratch",
        "mode": cfg.mode,
        "input_size": cfg.input_size,
        "pct": pct,
        "seed": seed,
        "n_train": int(len(y_tr)),
        "train_fingerprint": fingerprint(tr_paths),
        "n_params": n_params,
        "n_trainable": n_trainable,
        "feature_time_s": feature_time_s,
        "train_time_total_s": feature_time_s + result["train_time_s"],
        **result,
        "test_accuracy": test["accuracy"],
        "test_f1_macro": test["f1_macro"],
        "test_f1_per_class": dict(zip(CLASSES, test["f1_per_class"])),
        "test_confusion_matrix": test["confusion_matrix"],
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device_name(device),
        "torch": torch.__version__,
        "loop": cfg.loop.__dict__,
        "eval_limit": eval_limit,
    }


def result_path(out_dir: Path, cfg_name: str, pct: int, seed: int) -> Path:
    return out_dir / cfg_name / f"pct{pct:03d}_seed{seed}.json"


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(path)
