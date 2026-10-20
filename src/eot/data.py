"""Image loading and GPU-side preprocessing.

EuroSAT RGB is small: 27,000 x 64 x 64 x 3 bytes = 332 MB as uint8. I load it
once into a tensor and do resizing, augmentation and normalization on the
training device, batch by batch. On Kaggle this removes the CPU dataloader as a
bottleneck (two CPU cores there), and resizing 64 -> 224 on the GPU is
much faster than doing it per image with PIL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from eot.splits import CLASSES

# ImageNet statistics. The pretrained models expect them; the scratch CNN uses
# them too, which is harmless since it starts with BatchNorm anyway.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NATIVE_SIZE = 64
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}


def load_images(root: Path, paths: pd.Series) -> torch.Tensor:
    """Read images listed in `paths` (relative to `root`) as a uint8 tensor (N, 3, H, W)."""
    images = np.empty((len(paths), NATIVE_SIZE, NATIVE_SIZE, 3), dtype=np.uint8)
    for i, rel in enumerate(paths):
        with Image.open(root / rel) as img:
            arr = np.asarray(img.convert("RGB"))
        if arr.shape != (NATIVE_SIZE, NATIVE_SIZE, 3):
            raise ValueError(f"{rel}: expected 64x64 RGB, got {arr.shape}")
        images[i] = arr
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()


def labels_of(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor([CLASS_TO_INDEX[c] for c in df["label"]], dtype=torch.long)


def augment(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random flips and 90-degree rotations, drawn independently per image.

    A satellite tile has no natural "up", so these 8 symmetries are label
    preserving and add no interpolation artefacts.
    """
    n = x.shape[0]
    rand = torch.randint(0, 2, (n, 2), generator=generator, device="cpu").to(x.device).bool()
    rot = torch.randint(0, 4, (n,), generator=generator, device="cpu").to(x.device)
    x = torch.where(rand[:, 0, None, None, None], x.flip(-1), x)
    x = torch.where(rand[:, 1, None, None, None], x.flip(-2), x)
    out = x.clone()
    for k in (1, 2, 3):
        mask = rot == k
        if mask.any():
            out[mask] = torch.rot90(x[mask], k, dims=(-2, -1))
    return out


def preprocess(x_uint8: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 (N, 3, 64, 64) -> normalized float (N, 3, size, size)."""
    x = x_uint8.float().div_(255.0)
    if size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
