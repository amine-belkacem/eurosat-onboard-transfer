"""Data pipeline, models and training loop, on CPU with synthetic data.

No pretrained weights are downloaded (pretrained=False everywhere), so the
suite runs offline in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from eot import data, models, training
from eot.experiment import Dataset, ExperimentConfig, run
from eot.splits import CLASSES, list_images, make_split

TINY_LOOP = training.LoopConfig(batch_size=8, lr_head=1e-3, lr_backbone=None, weight_decay=0.0,
                                max_epochs=1, min_steps=4, n_evals=2, patience=2,
                                eval_batch_size=16, amp=False)


def check(got, expected, what=""):
    assert got == expected, f"ECHEC {what}: obtenu {got}, attendu {expected}"


def check_close(got, expected, what="", tol=1e-4):
    assert abs(got - expected) <= tol, f"ECHEC {what}: obtenu {got}, attendu {expected}"


# ---------------------------------------------------------------- schedule

@pytest.mark.parametrize("n, bs, epochs, min_steps, n_evals, total, evals", [
    # 1 %: ceil(191/64) = 3 steps/epoch, 15*3 = 45 < 600 -> 600 steps, eval every 600/15 = 40
    (191, 64, 15, 600, 15, 600, list(range(40, 601, 40))),
    # 100 %: ceil(18900/64) = 296 steps/epoch, 15*296 = 4440 > 600, eval every epoch
    (18900, 64, 15, 600, 15, 4440, list(range(296, 4441, 296))),
    # ceil(100/64) = 2, max(2, 6) = 6; round(1.5)=2, round(3)=3, round(4.5)=4, round(6)=6
    (100, 64, 1, 6, 4, 6, [2, 3, 4, 6]),
])
def test_schedule(n, bs, epochs, min_steps, n_evals, total, evals):
    check(training.schedule(n, bs, epochs, min_steps, n_evals), (total, evals), "schedule")


def test_warmup_cosine():
    # total 105, warm-up int(105*0.05) = 5 steps
    f = training.warmup_cosine(105, 0.05)
    check_close(f(0), 1 / 5, "debut du warm-up")
    check_close(f(4), 1.0, "fin du warm-up")
    check_close(f(5), 1.0, "debut du cosinus")
    check_close(f(55), 0.5, "mi-parcours: (55-5)/100 = 0.5 -> 0.5*(1+cos(pi/2))")
    check_close(f(105), 0.0, "fin")


# ---------------------------------------------------------------- data

def test_load_images(tmp_path):
    (tmp_path / "A").mkdir()
    Image.new("RGB", (64, 64), (200, 30, 90)).save(tmp_path / "A" / "A_1.png")
    x = data.load_images(tmp_path, pd.Series(["A/A_1.png"]))
    check(tuple(x.shape), (1, 3, 64, 64), "forme")
    check(x.dtype, torch.uint8, "type")
    check(x[0, :, 10, 10].tolist(), [200, 30, 90], "pixel RGB")


def test_load_images_rejects_wrong_size(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "small.png")
    with pytest.raises(ValueError):
        data.load_images(tmp_path, pd.Series(["small.png"]))


def test_augment_is_one_of_8_symmetries():
    g = torch.Generator().manual_seed(0)
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8).view(2, 3, 4, 4)
    y = data.augment(x, generator=g)
    for i in range(2):
        variants = []
        for base in (x[i], x[i].flip(-1)):
            variants += [torch.rot90(base, k, dims=(-2, -1)) for k in range(4)]
        check(any(torch.equal(y[i], v) for v in variants), True, f"image {i} est une symetrie")


def test_preprocess_values_and_size():
    x = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    out = data.preprocess(x, 224)
    check(tuple(out.shape), (1, 3, 224, 224), "taille apres resize")
    # black pixel, channel 0: (0 - 0.485) / 0.229 = -2.117904
    check_close(out[0, 0, 100, 100].item(), -0.485 / 0.229, "normalisation canal R")


# ---------------------------------------------------------------- models

def test_small_cnn_parameter_count():
    # block(3,32):   3*32*9 + 64 + 32*32*9 + 64      =  10 208
    # block(32,64):  32*64*9 + 128 + 64*64*9 + 128   =  55 552
    # block(64,128): 64*128*9 + 256 + 128*128*9 + 256 = 221 696
    # block(128,256): 128*256*9 + 512 + 256*256*9 + 512 = 885 760
    # head: 256*10 + 10                              =   2 570
    # total                                          = 1 175 786
    check(models.count_parameters(models.SmallCNN()), 1_175_786, "parametres SmallCNN")


@pytest.mark.parametrize("name, size", [("small_cnn", 64), ("resnet50", 224), ("vit_small", 224)])
def test_model_output_shape(name, size):
    model = models.create_model(name, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))
    check(tuple(out.shape), (2, 10), f"sortie {name}")


def test_head_parameters_are_the_new_classifier():
    model = models.create_model("resnet50", pretrained=False)
    shapes = [tuple(p.shape) for p in models.head_parameters(model)]
    check(shapes, [(10, 2048), (10,)], "parametres de la tete ResNet-50")


# ---------------------------------------------------------------- training loop

def _prep(xb, train):
    return data.preprocess(xb, xb.shape[-1])


def test_overfit_tiny_set():
    # 20 random images with arbitrary labels: a working loop must memorize them.
    torch.manual_seed(0)
    x = torch.randint(0, 256, (20, 3, 64, 64), dtype=torch.uint8)
    y = torch.arange(20) % 10
    model = models.SmallCNN()
    cfg = training.LoopConfig(batch_size=20, lr_head=3e-3, lr_backbone=None, weight_decay=0.0,
                              max_epochs=150, min_steps=0, n_evals=10, patience=10, amp=False)
    res = training.fit(model, models.head_parameters(model), x, y, x, y, _prep, cfg,
                       torch.device("cpu"), seed=0)
    check(res["best_val_acc"], 1.0, "precision sur 20 images apprises par coeur")


def test_frozen_backbone_untouched_by_probe():
    torch.manual_seed(0)
    backbone = models.create_backbone("resnet18", pretrained=False)
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    x = torch.randint(0, 256, (16, 3, 64, 64), dtype=torch.uint8)
    y = torch.arange(16) % 10
    feats = training.extract_features(backbone, x, _prep, torch.device("cpu"), 8, amp=False)
    head, params = training.linear_probe_head(feats, 10)
    training.fit(head, params, feats, y, feats, y, lambda f, t: f, TINY_LOOP, torch.device("cpu"), 0)
    changed = [k for k, v in backbone.state_dict().items() if not torch.equal(v, before[k])]
    check(changed, [], "tenseurs du backbone modifies (poids ou stats BatchNorm)")
    check(int(feats.shape[1]), 512, "dimension des features ResNet-18")


def test_finetune_updates_backbone():
    torch.manual_seed(0)
    model = models.create_model("resnet18", pretrained=False)
    w0 = model.conv1.weight.detach().clone()
    x = torch.randint(0, 256, (16, 3, 64, 64), dtype=torch.uint8)
    y = torch.arange(16) % 10
    cfg = training.LoopConfig(**{**TINY_LOOP.__dict__, "lr_backbone": 1e-4})
    training.fit(model, models.head_parameters(model), x, y, x, y, _prep, cfg, torch.device("cpu"), 0)
    check(torch.equal(model.conv1.weight.detach(), w0), False, "premiere couche modifiee")


# ---------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Dataset:
    root = tmp_path_factory.mktemp("eurosat")
    for label in CLASSES:
        (root / label).mkdir()
        for i in range(1, 21):
            (root / label / f"{label}_{i}.jpg").touch()
    df = make_split(list_images(root))  # 20 per class -> 14 / 3 / 3
    g = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (len(df), 3, 64, 64), dtype=torch.uint8, generator=g)
    return Dataset(df, images, data.labels_of(df))


@pytest.mark.parametrize("mode, lr_backbone", [("scratch", None), ("finetune", 1e-4), ("probe", None)])
def test_run_end_to_end(tiny_dataset, mode, lr_backbone):
    model = "small_cnn" if mode == "scratch" else "resnet18"
    cfg = ExperimentConfig(name=f"t_{mode}", model=model, mode=mode, input_size=64,
                           loop=training.LoopConfig(**{**TINY_LOOP.__dict__, "lr_backbone": lr_backbone}))
    res = run(cfg, tiny_dataset, pct=25, seed=1, device=torch.device("cpu"), pretrained=False)
    # 25 % of 14 training images per class: (14*25 + 50)//100 = 4 -> 40 images
    check(res["n_train"], 40, "taille du sous-ensemble 25 %")
    check(len(res["test_confusion_matrix"]), 10, "matrice de confusion 10x10")
    # 3 test images per class -> 30 in total
    check(int(np.sum(res["test_confusion_matrix"])), 30, "nombre d'images de test")
    check(0.0 <= res["test_accuracy"] <= 1.0, True, "precision dans [0, 1]")
    if mode == "probe":
        check(res["n_trainable"], 512 * 10 + 10, "parametres entrainables de la sonde")
