"""Model factory: the from-scratch CNN and the timm backbones."""

from __future__ import annotations

import timm
import torch
from torch import nn

NUM_CLASSES = 10

# timm names with their pretrained tags, so the exact weights are on record.
TIMM_MODELS = {
    "resnet50": "resnet50.a1_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "resnet18": "resnet18.a1_in1k",
}


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """VGG-style CNN for 64x64 inputs: four conv blocks (64 -> 4 px), then GAP and a linear head.

    About 1.2 M parameters. Deliberately plain: it is the "no transfer" reference,
    not an attempt at the best possible scratch model.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, widths: tuple[int, ...] = (32, 64, 128, 256),
                 dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers.append(_block(c_in, w))
            c_in = w
        self.features = nn.Sequential(*layers)
        self.num_features = c_in
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def create_model(name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Full classifier (backbone + new head) for scratch training or fine-tuning."""
    if name == "small_cnn":
        return SmallCNN(num_classes)
    if name not in TIMM_MODELS:
        raise ValueError(f"unknown model {name!r}; choose from small_cnn, {', '.join(TIMM_MODELS)}")
    return timm.create_model(TIMM_MODELS[name], pretrained=pretrained, num_classes=num_classes)


def create_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """Backbone returning pooled features (num_classes=0), for the linear probe."""
    if name not in TIMM_MODELS:
        raise ValueError(f"linear probe needs a pretrained timm model, got {name!r}")
    return timm.create_model(TIMM_MODELS[name], pretrained=pretrained, num_classes=0)


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters of the freshly initialised classifier, trained with a higher LR."""
    if isinstance(model, SmallCNN):
        return list(model.head.parameters())
    return list(model.get_classifier().parameters())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
