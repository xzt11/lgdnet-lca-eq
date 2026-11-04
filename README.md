# Land-Cover-Conditioned Post-Earthquake Damage Mapping from Very High Resolution Imagery

Official PyTorch implementation of LGDNet for land-cover-conditioned earthquake
damage assessment from post-event very-high-resolution RGB imagery.

This repository provides the code and release assets described in the paper:

- LGDNet / LSGM model code
- training configuration for the paper setting
- two-task loss implementation for land-cover and damage supervision
- preprocessing and evaluation scripts
- LCA-EQ labels, AOI metadata, event metadata, and the paper event-level split
- source-specific dataset access and licensing information
- link to the best checkpoint on Zenodo

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

The trained checkpoint is archived on Zenodo:

```text
Dataset DOI (LCA-EQ v1.1): https://doi.org/10.5281/zenodo.23934838
Dataset concept DOI:        https://doi.org/10.5281/zenodo.21472883
Model DOI:                  https://doi.org/10.5281/zenodo.21479452
```

Zenodo v1.1 retains the original v1.0 image archive and adds a supplement with
445 explicit Southwest Puerto Rico all-negative damage masks, so all 6,683
patches have a damage-mask file. Follow the Zenodo v1.1 `README_v1.1.md` and
`README_REASSEMBLE_DATASET_v1.1.txt` instructions to reconstruct the dataset.
The earlier v1.0 record remains an immutable historical version. The v1.1
record's Zenodo rights field is `Other (Non-Commercial)`; its description and
`LICENSE_DATA.md` distinguish author-created data, eligible Maxar assets, and
the separately permitted Google Earth imagery. Neither the GitHub MIT license
nor the CC BY-NC 4.0 terms relicense Google Earth pixels.

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

Land-cover classes used for loss and evaluation:

| Id | Class |
| --- | --- |
| 0 | Buildings |
| 1 | Roads |
| 2 | Impervious surface |
| 3 | Forest |
| 4 | Farmland |
| 5 | Water |
| 6 | Other |

The PNG masks use a source encoding that differs from the model's seven-class
order. The loader performs this mapping:

| Raw land-mask ID | Source class | Model ID |
| --- | --- | --- |
| 0 | Background / outside AOI | 255 (ignore) |
| 1 | Road | 1 |
| 2 | Farmland | 4 |
| 3 | Building | 0 |
| 4 | Forest | 3 |
| 5 | Impervious surface | 2 |
| 6 | Other | 6 |
| 7 | Water | 5 |

Damage masks encode `0` as no visible damage, `1` as visible damage, and `255`
as ignore where applicable. The original annotation masks are retained without
post-hoc relabeling. The training loader additionally ignores damage pixels
outside the annotated AOI by applying the land-mask validity region.

The 445 Southwest Puerto Rico training patches now each have an explicit
all-negative damage mask: valid AOI pixels are `0`, and pixels outside the AOI
are `255`. These files are included in `data/lca_eq/southwest_puerto_rico_2020/
damage_masks/` and are linked from the training manifest.

The paper defines Building, Road, and Impervious Surface as damage-support
classes. The released source labels are preserved as annotated rather than
rewritten in preprocessing; users should use the label version and evaluation
protocol associated with the reported paper results. The loader does not
silently change positive damage labels based on land-cover predictions.

The training config uses the seven-class paper order and LSGM support IDs
`[0, 1, 2]` (Building, Road, Impervious Surface); the four remaining classes
are non-support. This semantic grouping guides the model; the loader does not
rewrite the original damage annotation mask based on land-cover labels.

If released masks store background as `0` and foreground categories as `1..7`,
`LCAEQDataset` remaps them to the paper label order above and converts
background/outside-AOI pixels to `255`.

## Model

LGDNet uses a single post-event RGB image as input and predicts two outputs:

1. seven-class land-cover segmentation;
2. binary visible-damage segmentation.

The main network consists of an ImageNet-pretrained ResNet-101 encoder
(stage depths 3/4/23/3; feature strides 4/8/16/32), a two-layer, eight-head
Transformer bottleneck (448-dimensional tokens and 4x feed-forward width), and
an FPN decoder with three Mamba refinement blocks on the stride-16/8/4 top-down
features. The shared decoded feature feeds separate seven-class land-cover and
damage-feature branches. LSGM modulates the damage feature, which is then passed
to a two-logit Softmax damage classifier. The land-cover logits are upsampled
before LSGM; its grouped probabilities are bilinearly resized to the damage
feature grid. The configured LSGM projection maps 448 damage channels to a
224-dimensional embedding (`C'`).

For clarity, the ResNet-101 stages produce 256/512/1024/2048 channels at
strides 4/8/16/32. The stride-32 feature is projected to 448 channels and
processed by a 2-layer Transformer (8 heads, 1792-dimensional feed-forward
layer, GELU, pre-LayerNorm, dropout 0.1). FPN uses 1x1 lateral projections
from the stride-16/8/4 stages, bilinear top-down addition, and one Mamba block
at each of those three resolutions; two 3x3 Conv-BN-SiLU layers refine the
stride-4 output. Each task branch begins with a 3x3 Conv-BN-SiLU and Dropout2d
(0.1). The semantic branch ends in a 1x1 seven-class classifier. The damage
branch passes its 448-channel feature through LSGM and then a 1x1 two-logit
classifier for undamaged/damaged pixels. The damaged-class Softmax probability
is thresholded after upsampling. Each Mamba block uses pre-LayerNorm, `d_state=16`, `d_conv=4`,
expansion 2, and a learnable residual scale initialized to zero.

`MambaDecoderBlock` uses a selective state-space Mamba mixer over flattened
spatial tokens. The public code includes a PyTorch reference implementation with
the same Mamba parameterization (`A_log`, `D`, input-dependent `B/C/delta`, and
selective scan). For faster training/inference, install the optional optimized
Mamba kernels:

```bash
pip install -e ".[mamba]"
```

The manuscript reports the LGDNet efficiency as:

```text
parameters: 60.67 M
FLOPs:      505.95 G
FPS:        9.93
```

These are historical manuscript-reported values, retained here as reported
metadata. The original profiler outputs and timing script are unavailable, so
the current public code snapshot has not been independently profiled to verify
that it reproduces the manuscript's parameter, FLOPs, and FPS figures.

The Land-Semantics Gating Module (LSGM) groups land-cover probabilities into
support classes (`Buildings`, `Roads`, `Impervious surface`) and non-support
classes (`Forest`, `Farmland`, `Water`, `Other`). It pools semantic prototypes,
estimates normalized prototype affinities, and recalibrates damage features by
residual modulation:

```text
F_out = F_D * (1 + lambda_S * A_S - lambda_U * A_U)
```

The implementation uses a learnable 1x1 damage-feature projection, one
masked-average prototype for each semantic group, L2-normalized cosine
affinities followed by sigmoid/positive softplus temperatures, and
`lambda_S=sigmoid(theta_S)`, `lambda_U=0.9*sigmoid(theta_U)`. Semantic
probabilities remain differentiable through the gate. The configured numerical
stability epsilon is `1e-6`.

## Paper training setting

The default configuration follows the paper:

```text
random crop size: 384 x 384
epochs: 30
batch size: 8
optimizer: AdamW + Lookahead
pretrained encoder learning rate: 6e-5
new module learning rate: 6e-4
scheduler: cosine annealing
augmentation: scale jittering 0.75-1.25, horizontal/vertical flipping,
              per-channel ImageNet normalization
loss: L = 1.0 * L_damage + 0.5 * L_land
L_land: weighted cross-entropy + Dice
L_damage: weighted cross-entropy + Dice
damage output: two logits; damaged-class Softmax probability is thresholded;
              two-class Dice includes both undamaged and damaged classes
monitor: val_combined_F1
```

The released configuration computes training-set inverse-frequency weights
for both tasks as $w_c=N/(C n_c)$ over valid pixels in `train_manifest`; it
prints and stores the resulting values in each checkpoint. These values are
reproducible for the supplied masks and split, but are not asserted to recover
the unavailable numerical weights used in the historical run. The configured
seed is 42 for this executable reference run; it is not verified as the seed
used by every historical result. The training script uses FP32 (no autocast);
mixed-precision use in historical runs has not been verified.
Weighted cross-entropy is summed over valid pixels and divided
by the valid-pixel count; Dice uses epsilon `1e-6`, averages over all classes
(including the undamaged class), and excludes ignore label `255`. For damage,
the Puerto Rico hard-negative masks are explicitly included in the release and
train manifest as described in the dataset section. At each validation pass, a
pooled damage-F1 threshold is selected on a 0.001 grid; that
validation-selected threshold is saved with the checkpoint and must remain
fixed for test evaluation. The test split is never used for threshold
calibration.

Run training:

```bash
python scripts/train.py --config configs/lgdnet_lca_eq.yaml
```

Run evaluation (when no explicit threshold is supplied, test evaluation first
calibrates a threshold on the validation split):

```bash
python scripts/evaluate_lgdnet.py   --config configs/lgdnet_lca_eq.yaml   --checkpoint /path/to/best_model.ckpt   --split test
```

Prepare or validate local dataset paths:

```bash
python scripts/preprocess_lca_eq.py --root data/lca_eq
```

Recompute the training-set class weights:

```bash
python scripts/compute_lca_eq_class_weights.py --root data/lca_eq
```

## Checkpoint

Historical checkpoint metadata (not independently supported by original run logs):

```text
best epoch: 28
best val_combined_F1: 0.8286
historically reported damage threshold: 0.75 (not independently verified)
SHA256: 754a9d0c7a259c37e4b832dacafae2df72b72693f97707a2c3a101c779530ea9
```

The checkpoint is provided through the Zenodo model record above rather than
stored directly in Git. Its compatibility with this two-logit manuscript-aligned
implementation has not been verified here; the evaluator uses strict loading
and will reject incompatible weights rather than silently ignoring mismatched
parameters. The epoch, validation score, threshold, and SHA above are archival
metadata, not values independently recovered from the original run logs.
The exact training-label archive used for that historical checkpoint is not
independently verified. Treat the checkpoint metrics as historical reported
metadata, not as a fresh evaluation on a newly assembled dataset package.

## Data licensing and provenance

The root `LICENSE` applies to LGDNet source code only; it does not license
third-party imagery. Seven event sources are described in the manuscript as
Maxar Open Data Program imagery. CC BY-NC 4.0 applies only to source assets
covered by that program and is subject to its attribution and other terms.
Yangbi imagery was accessed through Google Earth historical imagery; its RGB
pixels are not redistributed in this GitHub repository, and the code license
does not grant rights to obtain or redistribute them. See
[`data/lca_eq/LICENSE_DATA.md`](data/lca_eq/LICENSE_DATA.md) for the component-
specific terms and attribution guidance. Zenodo v1.1 uses a source-specific
rights statement: CC BY-NC 4.0 for author-created data and eligible Maxar
assets, with Google Earth imagery included under separate author-obtained
permission and still subject to the applicable source terms.

## Reproducibility record for reviewer C10

See [`docs/C10_REPRODUCIBILITY.md`](docs/C10_REPRODUCIBILITY.md) for the full
architecture and training specification, executable commands, and an explicit
account of details that remain unavailable for the historical experiments.

## Repository layout

```text
configs/
data/lca_eq/
scripts/preprocess_lca_eq.py
scripts/train.py
scripts/evaluate_lgdnet.py
src/lgdnet/
tests/
```

## License

MIT License. Users are responsible for complying with the licensing terms of the
source imagery used in the full LCA-EQ archive.
