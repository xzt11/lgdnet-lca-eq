# LGDNet

LGDNet maps land cover and visible earthquake damage from post-event, very-high-resolution RGB imagery. Its land-cover predictions guide damage-feature refinement through the Land-Semantics Gating Module (LSGM).

## Repository

This repository contains the PyTorch model, training and evaluation scripts, configuration, and lightweight LCA-EQ metadata and split files. The complete dataset and model checkpoint are archived on Zenodo:

- [LCA-EQ dataset v1.1](https://doi.org/10.5281/zenodo.23934838)
- [LCA-EQ dataset concept record](https://doi.org/10.5281/zenodo.21472883)
- [LGDNet checkpoint](https://doi.org/10.5281/zenodo.21479452)

Follow the dataset instructions on Zenodo to obtain and assemble the data. Image access and reuse are subject to the source-specific terms described in [`data/lca_eq/LICENSE_DATA.md`](data/lca_eq/LICENSE_DATA.md); the GitHub software license does not grant rights to third-party imagery.

## Quick start

Requires Python 3.10 or later and PyTorch. Install the project with:

```bash
pip install -e ".[dev]"
```

Train or evaluate using the provided configuration:

```bash
python scripts/train.py --config configs/lgdnet_lca_eq.yaml
python scripts/evaluate_lgdnet.py --config configs/lgdnet_lca_eq.yaml --checkpoint /path/to/checkpoint.ckpt --split test
```

The default configuration uses the paper's seven land-cover classes and event-level train/validation/test split. See [`data/lca_eq/README.md`](data/lca_eq/README.md) for dataset files and [`docs/C10_REPRODUCIBILITY.md`](docs/C10_REPRODUCIBILITY.md) for implementation and experiment details.

## License

The source code is released under the MIT License. Dataset components retain their respective source and licensing terms.
