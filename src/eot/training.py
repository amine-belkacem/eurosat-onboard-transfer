"""Training loop shared by the three regimes: scratch, fine-tuning and linear probe.

Budget in steps, not epochs. At 1 % of the data an epoch is 3 batches, so
"15 epochs" would mean 45 optimizer steps, which is too few to compare with
15 epochs at 100 % (4,440 steps). Each run therefore trains for
max(max_epochs epochs, min_steps steps) and is evaluated on validation at
n_evals evenly spaced points. The weights with the best validation accuracy
are kept, and the run stops early after `patience` evaluations without
improvement.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

Prepare = Callable[[torch.Tensor, bool], torch.Tensor]


@dataclass(frozen=True)
class LoopConfig:
    batch_size: int
    lr_head: float
    lr_backbone: float | None  # None: one learning rate for the whole model
    weight_decay: float
    max_epochs: int
    min_steps: int
    n_evals: int
    patience: int
    warmup_frac: float = 0.05
    eval_batch_size: int = 256
    amp: bool = True


def schedule(n_train: int, batch_size: int, max_epochs: int, min_steps: int,
             n_evals: int) -> tuple[int, list[int]]:
    """Total number of optimizer steps and the steps after which validation runs."""
    if n_train <= 0 or batch_size <= 0 or n_evals <= 0:
        raise ValueError("n_train, batch_size and n_evals must be positive")
    steps_per_epoch = math.ceil(n_train / batch_size)
    total = max(max_epochs * steps_per_epoch, min_steps)
    evals = sorted({max(1, round(total * (i + 1) / n_evals)) for i in range(n_evals)})
    return total, evals


def warmup_cosine(total_steps: int, warmup_frac: float) -> Callable[[int], float]:
    """LR multiplier: linear warm-up, then cosine decay to zero."""
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _batches(n: int, batch_size: int, generator: torch.Generator):
    """Endless stream of shuffled index batches, reshuffled every epoch."""
    while True:
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            yield perm[start:start + batch_size]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, prepare: Prepare, device: torch.device,
            batch_size: int = 256, amp: bool = True) -> np.ndarray:
    """Predicted class index for every row of `x` (kept on CPU, moved batch by batch)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    preds = []
    for start in range(0, x.shape[0], batch_size):
        xb = prepare(x[start:start + batch_size].to(device, non_blocking=True), False)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            preds.append(model(xb).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


@torch.no_grad()
def extract_features(backbone: nn.Module, x: torch.Tensor, prepare: Prepare, device: torch.device,
                     batch_size: int = 256, amp: bool = True) -> torch.Tensor:
    """Pooled features of a frozen backbone, as float32 on CPU."""
    backbone.eval()
    use_amp = amp and device.type == "cuda"
    feats = []
    for start in range(0, x.shape[0], batch_size):
        xb = prepare(x[start:start + batch_size].to(device, non_blocking=True), False)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            feats.append(backbone(xb).float().cpu())
    return torch.cat(feats)


def fit(model: nn.Module, head_params: list[nn.Parameter], x_train: torch.Tensor, y_train: torch.Tensor,
        x_val: torch.Tensor, y_val: torch.Tensor, prepare: Prepare, cfg: LoopConfig,
        device: torch.device, seed: int) -> dict:
    """Train `model` and return the best validation weights plus timing and history.

    `prepare(batch, train)` turns a raw batch (uint8 images or cached features)
    into model input on `device`; with train=True it may augment.
    """
    model.to(device)
    head_ids = {id(p) for p in head_params}
    others = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    if cfg.lr_backbone is None:
        groups = [{"params": list(model.parameters()), "lr": cfg.lr_head}]
    else:
        groups = [{"params": head_params, "lr": cfg.lr_head},
                  {"params": others, "lr": cfg.lr_backbone}]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    total_steps, eval_steps = schedule(len(y_train), cfg.batch_size, cfg.max_epochs,
                                       cfg.min_steps, cfg.n_evals)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, cfg.warmup_frac))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    y_val_np = y_val.numpy()

    best = {"val_acc": -1.0, "step": 0, "state": None, "time_s": 0.0}
    history, evals_without_gain = [], 0
    eval_set = set(eval_steps)
    _sync(device)
    t0 = time.perf_counter()
    model.train()
    for step, idx in enumerate(_batches(len(y_train), cfg.batch_size, generator), start=1):
        xb = prepare(x_train[idx].to(device, non_blocking=True), True)
        yb = y_train[idx].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step in eval_set:
            val_acc = float((predict(model, x_val, prepare, device, cfg.eval_batch_size, cfg.amp)
                             == y_val_np).mean())
            model.train()
            _sync(device)
            elapsed = time.perf_counter() - t0
            history.append({"step": step, "loss": loss.item(), "val_acc": val_acc, "time_s": elapsed})
            if val_acc > best["val_acc"]:
                best.update(val_acc=val_acc, step=step, time_s=elapsed,
                            state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
                evals_without_gain = 0
            else:
                evals_without_gain += 1
            if evals_without_gain >= cfg.patience:
                break
        if step >= total_steps:
            break
    _sync(device)
    return {
        "best_state": best["state"],
        "best_val_acc": best["val_acc"],
        "best_step": best["step"],
        "time_to_best_s": best["time_s"],
        "train_time_s": time.perf_counter() - t0,
        "steps_done": step,
        "total_steps_planned": total_steps,
        "stopped_early": step < total_steps,
        "history": history,
    }


class Standardize(nn.Module):
    """Feature standardization with statistics frozen from the training subset."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def linear_probe_head(train_features: torch.Tensor, num_classes: int) -> tuple[nn.Module, list[nn.Parameter]]:
    """Standardize + Linear, the standardization fitted on the training features only."""
    linear = nn.Linear(train_features.shape[1], num_classes)
    head = nn.Sequential(Standardize(train_features.mean(0), train_features.std(0)), linear)
    return head, list(linear.parameters())
