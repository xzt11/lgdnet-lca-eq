# LGDNet: Land-Guided Damage Network

Official PyTorch implementation and release index for **land-cover-conditioned post-earthquake damage mapping** from single-temporal very-high-resolution (VHR) post-event RGB imagery.

This repository accompanies the paper:

> Land-cover-conditioned post-earthquake damage mapping from single-temporal VHR imagery

The paper introduces the **Land-Cover Anchored Earthquake (LCA-EQ)** benchmark and **LGDNet**, a dual-head segmentation framework that predicts:

- a seven-class land-cover layer: `Building`, `Road`, `Impervious Surface`, `Forest`, `Farmland`, `Water`, `Other`
- an independent binary visible-damage layer

Visible damage is treated as a **state associated with an underlying land-cover class**, rather than as an eighth mutually exclusive land-cover category. LGDNet uses predicted land-cover probabilities as soft semantic guidance through the **Land-Semantics Gating Module (LSGM)**, promoting damage-related features on plausible built-environment support surfaces and suppressing damage-like responses on non-support surfaces.

## Repository Contents

This repository provides:

- LGDNet and LSGM implementation
- manifest-based LCA-EQ data loader
- manuscript-aligned training configuration
- loss and metric utilities
- event-level split files and event metadata
- data-release structure for derived LCA-EQ research assets
- method and data documentation

The original VHR image tiles are **not redistributed** because they are governed by third-party imagery licence terms. Reproducibility is supported through releasable derived assets, including land-cover labels, visible-damage annotations, AOI footprints, acquisition metadata, crop grids, split files, preprocessing/evaluation scripts, configuration files, and model code. Users must obtain lawful access to the corresponding source imagery to reconstruct the exact image inputs.

## LCA-EQ Data Summary

LCA-EQ contains **6,683 annotated $1024 \times 1024$ image patches** extracted from post-event VHR RGB imagery at an approximate ground sampling distance of **0.3--0.5 m**. The benchmark covers eight earthquake events between 2018 and 2025. Each image patch is paired with spatially aligned land-cover and visible-damage labels.

The benchmark uses an event-level split to evaluate cross-event generalization. All patches from the same earthquake event are assigned to the same subset. The split metadata is provided under `data/lca_eq/splits/`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

For Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The manuscript experiments use PyTorch 2.7.1. The package metadata requires `torch>=2.7.1`.

## Data Format

Create CSV manifests with the following columns:

```csv
image,land_mask,damage_mask,event
data/images/sample_001.png,data/land_masks/sample_001.png,data/damage_masks/sample_001.png,Turkey-Syria-2023
```

Expected masks:

- land-cover mask: single-channel integer image with values `0..6`
- damage mask: single-channel binary image with values `0` or `1`

Class ids:

| Id | Class |
| --- | --- |
| 0 | Building |
| 1 | Road |
| 2 | Impervious Surface |
| 3 | Forest |
| 4 | Farmland |
| 5 | Water |
| 6 | Other |

Damage is stored in a separate binary layer and should not be treated as a land-cover class.

## Quick Model Check

```python
import torch
from lgdnet.models import LGDNet

model = LGDNet(num_land_classes=7)
x = torch.randn(2, 3, 384, 384)
out = model(x)

print(out["land_logits"].shape)
print(out["damage_logits"].shape)
```

## Training Configuration

The manuscript-aligned configuration is:

```bash
configs/lgdnet_lca_eq.yaml
```

Key settings:

- PyTorch: 2.7.1
- random crop size: 384 x 384
- epochs: 30
- optimizer: AdamW
- objective: `1.0 * L_damage + 0.5 * L_land`
- no auxiliary damage loss is used in the manuscript configuration

Run training after preparing the manifests and masks:

```bash
python scripts/train.py --config configs/lgdnet_lca_eq.yaml
```

## Project Layout

```text
.
├── configs/
├── data/lca_eq/
│   ├── metadata/
│   ├── splits/
│   ├── crop_grids/
│   ├── aoi_footprints/
│   └── labels/
├── docs/
├── scripts/
├── src/lgdnet/
│   ├── data/
│   ├── models/
│   ├── losses.py
│   └── metrics.py
├── tests/
├── CITATION.cff
├── LICENSE
└── README.md
```

## Citation

If this repository helps your work, please cite the original paper and this codebase. See `CITATION.cff`.
