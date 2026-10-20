# EuroSAT onboard transfer

[![tests](https://github.com/amine-belkacem/eurosat-onboard-transfer/actions/workflows/tests.yml/badge.svg)](https://github.com/amine-belkacem/eurosat-onboard-transfer/actions/workflows/tests.yml)

Which transfer-learning strategy should you pick to classify Sentinel-2 image tiles when you have few labelled images and a small compute budget on board? I compare a CNN trained from scratch, a pretrained ResNet-50 and a pretrained ViT-S/16, each as a linear probe and fully fine-tuned, on three axes:

1. accuracy;
2. how many labelled images it needs (1 % to 100 % of the training set);
3. compute cost, both for training (Kaggle GPU) and for inference on a Raspberry Pi 5 standing in for an onboard computer.

> **Status: milestone 0 (data pipeline).** No model has been trained yet, so this README contains no accuracy figures. Results will be added as each milestone is measured.

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

## Reproduce

```bash
python -m venv .venv            # Python 3.12
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/pip install -e .
python scripts/download_eurosat.py              # ~95 MB, MD5-checked
python scripts/prepare_splits.py                # regenerates splits/eurosat_rgb.csv (identical)
python scripts/plot_dataset.py                  # figures/dataset_*.png
pytest
```

## Tests

`pytest` checks the split against values worked out by hand in the test comments: per-class sizes, rounding of the subset sizes (17.5 images rounds up to 18), nesting, disjointness, determinism and pinned fingerprints. The tests run on synthetic folders plus the committed CSV, so CI does not need to download the dataset.

## Roadmap

| Milestone | Content | State |
|---|---|---|
| M0 | Download, frozen split, nested subsets, tests, CI | done |
| M1 | Training loop; scratch CNN and ResNet-50 (probe and fine-tune) at 100 %, 3 seeds | next |
| M2 | ViT-S/16, all 5 fractions, 75 runs; accuracy, data-efficiency and training-time figures | |
| M3 | ONNX export, INT8 quantization, Raspberry Pi 5 latency and memory; accuracy/latency Pareto front | |
| M4 | 13-band multispectral version: adapting RGB-pretrained weights to 13 channels | |
| M5 | Geographic vs. random split (spatial leakage) | |
| M6 | Energy per image on the Pi with a USB-C power meter | |

A note for M5: I checked that the multispectral GeoTIFFs carry their georeferencing (UTM zone, tie point and 10 m pixel scale in the GeoTIFF tags), so a geographic split is possible. I still have to confirm that `Label_i.jpg` and `Label_i.tif` are the same tile before reusing the coordinates for the RGB images.

## References

- P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
- Contains modified Copernicus Sentinel data.

## License

Code under the MIT license (see [LICENSE](LICENSE)). EuroSAT is distributed by its authors under the MIT license.
