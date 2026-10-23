# EuroSAT onboard transfer

[![tests](https://github.com/amine-belkacem/eurosat-onboard-transfer/actions/workflows/tests.yml/badge.svg)](https://github.com/amine-belkacem/eurosat-onboard-transfer/actions/workflows/tests.yml)

Which transfer-learning strategy should you pick to classify Sentinel-2 image tiles when you have few labelled images and a small compute budget on board? I compare a CNN trained from scratch, a pretrained ResNet-50 and a pretrained ViT-S/16, each as a linear probe and fully fine-tuned, on three axes:

1. accuracy;
2. how many labelled images it needs (1 % to 100 % of the training set);
3. compute cost, both for training (Kaggle GPU) and for inference on a Raspberry Pi 5 standing in for an onboard computer.

> **Status: milestone 2 done**: 5 configs × 5 data fractions × 3 seeds, 75 training runs on a Kaggle T4. The Raspberry Pi measurements (milestone 3) come next, so nothing below says anything yet about inference cost on board.

## Results: accuracy against labelled data (milestone 2)

![Accuracy and training time against the number of labelled images](figures/data_efficiency.png)

Test accuracy (%), mean ± sample std over 3 seeds, 4,050 test images:

| Labelled images | 191 (1 %) | 947 (5 %) | 1,890 (10 %) | 4,727 (25 %) | 18,900 (100 %) |
|---|---|---|---|---|---|
| CNN from scratch (1.2 M params) | 77.48 ± 2.49 | 91.22 ± 0.77 | 93.33 ± 0.71 | 96.49 ± 0.25 | 98.63 ± 0.08 |
| ResNet-50 linear probe | 82.16 ± 1.22 | 91.06 ± 0.55 | 92.88 ± 0.45 | 94.63 ± 0.26 | 96.21 ± 0.25 |
| ResNet-50 fine-tuned (23.5 M) | 88.48 ± 2.91 | 95.28 ± 0.13 | 95.79 ± 0.55 | 97.50 ± 0.16 | 98.65 ± 0.20 |
| ViT-S/16 linear probe | 88.02 ± 1.00 | 93.26 ± 0.21 | 94.36 ± 0.28 | 95.68 ± 0.18 | 97.10 ± 0.04 |
| ViT-S/16 fine-tuned (21.7 M) | **92.84 ± 1.68** | **97.41 ± 0.36** | **98.04 ± 0.10** | **98.54 ± 0.23** | **99.14 ± 0.20** |

The full table with macro F1 and training times is in [`results/summary.md`](results/summary.md); every number comes from the per-run JSON files in [`results/`](results/). For a given fraction and seed, all five configs saw exactly the same training images.

What I take from it:

- **Transfer matters when labels are scarce, not when they are plentiful.** With 191 labelled images the scratch CNN reaches 77.5 %, the fine-tuned ViT 92.8 %: 15 points apart. With all 18,900 images the CNN catches up with the fine-tuned ResNet-50 (98.63 vs. 98.65) and sits half a point under the ViT.
- **Fine-tuned ViT-S is ahead at every fraction**, and at 100 % its three seeds (98.94, 99.14, 99.33) are all above the best ResNet-50 seed (98.84). It reaches with 947 labels (97.41 %) what ResNet-50 needs about 4,700 for (97.50 %), and it trained faster than ResNet-50 on the T4 (14.4 vs. 21.3 min at 100 %). I have not profiled why; my guess is that its large matrix multiplies suit the T4's half-precision units better than ResNet-50's convolutions.
- **The cheapest option is surprisingly good.** The ViT linear probe trains in about 6 seconds at 1 % (feature extraction included) and matches the fine-tuned ResNet-50 (88.0 vs. 88.5 %), which needs almost 4 minutes and has three times the seed-to-seed spread. It stays ahead of the scratch CNN up to 10 % of the data; from 25 % on, the CNN wins.
- **A caveat on "ViT vs. ResNet"**: the ViT weights I use were pretrained on ImageNet-21k (14 M images) before ImageNet-1k, while the ResNet-50 weights only saw ImageNet-1k. The gap therefore mixes architecture and pretraining data, and this study cannot separate the two.
- At 1 % the spread between seeds is large (up to ± 2.9 points). Part of it is which 191 images happened to be labelled, since each seed draws its own subset. Comparisons at that fraction need more seeds before reading anything into differences of a point or two.

### Full-data comparison

![Accuracy and training time at 100 %](figures/accuracy_time_pct100.png)

Frozen ImageNet features fall short mostly on River vs. Highway: about 50 errors each way over the 3 seeds for the ResNet-50 probe (40 and 25 for the ViT probe), plus the three crop and vegetation classes. Rivers and highways are both thin lines crossing the tile. Without fine-tuning, features learnt on ImageNet photos see the shape but do not weigh the colour and texture that separate water from asphalt. Fine-tuning ResNet-50 brings River → Highway down from 50 to 11. As a sanity check, the fine-tuned ResNet-50 lands close to the 98.57 % that Helber et al. report for it, on an 80/20 split, so the comparison is only loose.

![Confusion matrices](figures/confusion_resnet50_pct100.png)

### Limits

- 3 seeds give a rough spread, not a confidence interval.
- Hyperparameters were set once, before the first run, and are not tuned for any model. A tuned probe or a longer schedule for the CNN could move some lines by a point or so.
- Model selection uses the full validation set (4,050 labelled images) even when only 191 training images are labelled. A real low-label project would not have that many validation labels, so the 1 % numbers are on the optimistic side for every method alike.
- Training times at 1 to 10 % are dominated by the fixed minimum budget (600 to 1,500 steps) and by the 15 to 20 validation passes, which is why they barely change across those fractions.
- Training time says nothing about inference cost on board. That is milestone 3.

## Why this question

A satellite cannot downlink everything it captures. Classifying tiles on board and sending down only the useful ones saves bandwidth, as ESA showed with the onboard cloud filter of Φ-sat-1. On board, though, every millisecond and every megabyte costs something. Most EuroSAT notebooks stop at "fine-tuned model X reaches 98 %". I want to know what that accuracy costs in labelled data and in latency, and whether the ranking still holds once the model has to run on a small ARM CPU.

## Data

[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019), RGB version: 27,000 Sentinel-2 tiles of 64 × 64 px at 10 m/px, 10 land-use classes. MIT license, downloaded from the authors' [Zenodo record](https://zenodo.org/records/7711810) and checked against its MD5.

![EuroSAT examples](figures/dataset_examples.png)

### Split

The original paper uses an 80/20 split with no validation set. I use a **stratified 70 / 15 / 15 split**, so epochs and hyperparameters are chosen on validation and the test set is only read at the end. The split is drawn once (seed 0) and committed in [`splits/eurosat_rgb.csv`](splits/eurosat_rgb.csv). A test pins its SHA-256 fingerprint, so a split that drifts would fail CI.

![Split per class](figures/dataset_split.png)

### Low-data subsets

Each class has its own random ranking of its training images, one ranking per seed. A subset at p % keeps the first `round(p/100 · n_class)` images of each class, rounded half up. Two consequences:

- subsets are **nested** (the 1 % subset sits inside the 5 % one, and so on), so a gain between two fractions comes from adding data, not from a luckier draw;
- each of the 3 seeds labels **different** images, so the spread I report also covers "which images happened to be labelled". At 1 % (19 images per class on average) that is a large part of the variance.

| Fraction | 1 % | 5 % | 10 % | 25 % | 100 % |
|---|---|---|---|---|---|
| Training images | 191 | 947 | 1,890 | 4,727 | 18,900 |

## Training setup

| Config | Model | What is trained | Input | Budget |
|---|---|---|---|---|
| `cnn_scratch` | 4-block VGG-style CNN, 1.18 M params | everything, from random init | 64 px | max(40 epochs, 1,500 steps) |
| `resnet50_probe` | ResNet-50, ImageNet (`resnet50.a1_in1k`), frozen | a linear layer on pooled features | 224 px | max(100 epochs, 1,000 steps) |
| `resnet50_ft` | same weights | everything; head LR 1e-3, backbone 1e-4 | 224 px | max(15 epochs, 600 steps) |

Some choices that matter:

- **Steps, not epochs.** At 1 % an epoch is 3 batches, so "15 epochs" would be 45 optimizer steps. Each run trains for the larger of an epoch budget and a step budget, is evaluated on validation at evenly spaced points, keeps the best weights and stops after a few evaluations without gain.
- **Preprocessing on the GPU.** The whole dataset fits in 332 MB as uint8, so it is loaded once. Resizing to 224 px (bicubic), flips and 90° rotations and normalization run on the GPU batch by batch. Kaggle gives two CPU cores, which would otherwise be the bottleneck.
- **Linear probe on cached features.** Each probe run extracts features for its training subset and the validation set once (that time is counted in its training time), then fits `Standardize + Linear` on them. No augmentation for the probe.
- **Test set read once**, after model selection, with a timestamp stored in the result file.
- Hyperparameters were set before the first run and are not tuned. AdamW, linear warm-up over 5 % of the steps, then cosine decay; mixed precision on GPU.

## Reproduce

```bash
python -m venv .venv            # Python 3.12
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/pip install -e .
python scripts/download_eurosat.py              # ~95 MB, MD5-checked
python scripts/prepare_splits.py                # regenerates splits/eurosat_rgb.csv (identical)
python scripts/plot_dataset.py                  # figures/dataset_*.png
pytest
# quick end-to-end check on CPU (tiny budget, 200 eval images, results/_smoke)
python scripts/run_experiment.py --config configs/resnet50_probe.yaml --pcts 1 --seeds 0 --smoke
```

The real runs go through [`kaggle/milestone1.ipynb`](kaggle/milestone1.ipynb) on a Kaggle T4: it clones this repository, downloads the data, runs the configs and zips one JSON per run. `python scripts/summarize.py` then builds the tables from those files.

## Tests

`pytest` checks the split against values worked out by hand in the test comments: per-class sizes, rounding of the subset sizes (17.5 images rounds up to 18), nesting, disjointness, determinism and pinned fingerprints. The tests run on synthetic folders plus the committed CSV, so CI does not need to download the dataset.

## Roadmap

| Milestone | Content | State |
|---|---|---|
| M0 | Download, frozen split, nested subsets, tests, CI | done |
| M1 | Training loop; scratch CNN and ResNet-50 (probe and fine-tune) at 100 %, 3 seeds | done |
| M2 | ViT-S/16, all 5 fractions, 75 runs; accuracy, data-efficiency and training-time figures | done |
| M3 | ONNX export, INT8 quantization, Raspberry Pi 5 latency and memory; accuracy/latency Pareto front | next |
| M4 | 13-band multispectral version: adapting RGB-pretrained weights to 13 channels | |
| M5 | Geographic vs. random split (spatial leakage) | |
| M6 | Energy per image on the Pi with a USB-C power meter | |

A note for M5: I checked that the multispectral GeoTIFFs carry their georeferencing (UTM zone, tie point and 10 m pixel scale in the GeoTIFF tags), so a geographic split is possible. [`scripts/probe_geotiff.py`](scripts/probe_geotiff.py) redoes the check: it reads a few files out of the 2.1 GB archive with HTTP range requests and prints their tags. I still have to confirm that `Label_i.jpg` and `Label_i.tif` are the same tile before reusing the coordinates for the RGB images.

## References

- P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
- Contains modified Copernicus Sentinel data.

## License

Code under the MIT license (see [LICENSE](LICENSE)). EuroSAT is distributed by its authors under the MIT license.
