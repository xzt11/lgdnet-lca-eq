# LGDNet: Land-Guided Damage Network

Official PyTorch implementation of LGDNet for land-cover-conditioned earthquake
damage assessment from post-event very-high-resolution RGB imagery.

This repository provides the code and release assets described in the paper:

- LGDNet / LSGM model code
- training configuration for the paper setting
- two-task loss implementation for land-cover and damage supervision
- preprocessing and evaluation scripts
- LCA-EQ labels, AOI metadata, event metadata, and the paper scene split
- links to the full LCA-EQ image archive and the best checkpoint on Zenodo

## Environment

The paper experiments use:

```text
Python >= 3.10
PyTorch 2.7.1
torchvision 0.22.1
```

Install:

```bash
pip install -e ".[dev]"
```

## Dataset

The GitHub repository contains lightweight LCA-EQ release assets under:

```text
data/lca_eq/
```

The full image archive and trained checkpoint are archived on Zenodo and should
be used for full reproduction:

```text
Dataset DOI: https://doi.org/10.5281/zenodo.21472884
Model DOI:   https://doi.org/10.5281/zenodo.21479452
```

Expected LCA-EQ layout after restoring the full dataset:

```text
data/lca_eq/
├── events.csv
├── aoi/
├── splits/paper_scene_split/{train,val,test}.csv
├── <event_id>/images/
├── <event_id>/land_masks/
└── <event_id>/damage_masks/
```

Paper scene split:

```text
train: 5490
val: 353
test: 840
```

Land-cover classes:

| Id | Class |
| --- | --- |
| 0 | Buildings |
| 1 | Roads |
| 2 | Impervious surface |
| 3 | Forest |
| 4 | Farmland |
| 5 | Water |
| 6 | Other |

Damage masks are binary. Samples without visible-damage masks use damage label
`255` and are ignored for damage supervision.

## Model

LGDNet uses a single post-event RGB image as input and predicts two outputs:

1. seven-class land-cover segmentation;
2. binary visible-damage segmentation.

The Local-Semantic Guided Module (LSGM) groups land-cover probabilities into
support classes (`Buildings`, `Roads`, `Impervious surface`) and non-support
classes (`Forest`, `Farmland`, `Water`, `Other`). It pools semantic prototypes,
estimates normalized prototype affinities, and recalibrates damage features by
residual modulation:

```text
F_out = F_D * (1 + lambda_S * A_S - lambda_U * A_U)
```

## Paper training setting

The default configuration follows the paper:

```text
random crop size: 384 x 384
epochs: 30
loss: L = lambda_land * L_land + lambda_damage * L_damage
L_land: weighted cross-entropy + Dice
L_damage: weighted cross-entropy + Dice
monitor: val_combined_F1
```

Run training:

```bash
python scripts/train.py --config configs/lgdnet_lca_eq.yaml
```

Run evaluation:

```bash
python scripts/evaluate_lgdnet.py   --config configs/lgdnet_lca_eq.yaml   --checkpoint /path/to/best_model.ckpt   --split test
```

Prepare or validate local dataset paths:

```bash
python scripts/preprocess_lca_eq.py --root data/lca_eq
```

## Checkpoint

Best alltrain checkpoint recorded for the paper setting:

```text
best epoch: 28
best val_combined_F1: 0.8286
recommended damage threshold: 0.75
SHA256: 754a9d0c7a259c37e4b832dacafae2df72b72693f97707a2c3a101c779530ea9
```

The checkpoint is provided through the Zenodo model record above rather than
stored directly in Git.

## Repository layout

```text
configs/
data/lca_eq/
experiments/
scripts/preprocess_lca_eq.py
scripts/train.py
scripts/evaluate_lgdnet.py
src/lgdnet/
tests/
```

## License

MIT License. Users are responsible for complying with the licensing terms of the
source imagery used in the full LCA-EQ archive.
