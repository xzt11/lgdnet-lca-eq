# LGDNet: Land-Guided Damage Network

Open-source PyTorch implementation scaffold for **land-cover-conditioned earthquake damage assessment** from very-high-resolution post-event RGB imagery.

This repository is based on the paper:

> Resolving Visual-Semantic Entanglement in VHR Damage Assessment: A Land-Cover Conditioned Framework

The paper introduces the **LCA-EQ** benchmark and **LGDNet**, a dual-head segmentation framework that predicts:

- a seven-class land-cover layer: `Buildings`, `Roads`, `Impervious Surfaces`, `Forest`, `Farmland`, `Water`, `Others`
- a binary earthquake damage-state layer

LGDNet uses land-cover probabilities as host priors through the **Land-Semantics Gating Module (LSGM)**, suppressing damage responses in non-host regions and promoting responses on plausible anthropogenic hosts.

## Repository Status

This is a clean open-source project scaffold intended for reproducibility and further development. It includes:

- LGDNet model components
- LSGM implementation
- dataset manifest loader
- training configuration template
- metric utilities
- paper-derived method notes

The LCA-EQ dataset is not included. Add your own data according to the format below.

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
| 0 | Buildings |
| 1 | Roads |
| 2 | Impervious Surfaces |
| 3 | Forest |
| 4 | Farmland |
| 5 | Water |
| 6 | Others |

## Quick Model Check

```python
import torch
from lgdnet.models import LGDNet

model = LGDNet(num_land_classes=7)
x = torch.randn(2, 3, 512, 512)
out = model(x)

print(out["land_logits"].shape)
print(out["damage_logits"].shape)
```

## Training

The paper reports training with PyTorch 2.0, AdamW, 100 epochs, 512 x 512 random crops, cosine annealing, and a composite loss:

```text
L_total = 1.0 * L_damage + 0.5 * L_land + 0.3 * L_aux
```

Damage supervision uses strong positive weighting because damaged pixels are sparse.

Configuration template:

```bash
configs/lgdnet_lca_eq.yaml
```

## Project Layout

```text
.
├── configs/
├── docs/
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

If this repository helps your work, cite the original paper and this codebase. See `CITATION.cff`.

## License

MIT License. See `LICENSE`.
