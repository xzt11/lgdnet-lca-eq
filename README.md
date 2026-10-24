# Land-Cover-Conditioned Post-Earthquake Damage Mapping from Very High Resolution Imagery

Official PyTorch implementation of LGDNet for land-cover-conditioned earthquake
damage assessment from post-event very-high-resolution RGB imagery.

This repository provides the code and release assets described in the paper:

- LGDNet / LSGM model code
- training configuration for the paper setting
- two-task loss implementation for land-cover and damage supervision
- preprocessing and evaluation scripts
- LCA-EQ labels, AOI metadata, event metadata, and the paper event-level split
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

Background pixels outside annotated AOIs are encoded as `255` after preprocessing
and are excluded from land-cover loss and evaluation. Damage masks are binary.
Samples without visible-damage masks use damage label `255` and are ignored for
damage supervision.

`southwest_puerto_rico_2020` is the hard-negative event used in the paper: it
contains no positive visible-damage labels, and missing damage masks for this
event are loaded as valid all-zero masks so that the patches contribute to
damage supervision.

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
to a one-logit sigmoid damage classifier. The land-cover logits are upsampled
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
branch passes its 448-channel feature through LSGM and then a 1x1 one-logit
classifier. Each Mamba block uses pre-LayerNorm, `d_state=16`, `d_conv=4`,
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

These are historical manuscript values; the original profiler outputs and
timing script have not been recovered. Counting trainable and non-trainable
parameters in the current public model definition gives 60.30 M, so it does
not exactly reproduce the reported 60.67 M. FLOPs and FPS have not been
independently measured for this source snapshot. The YAML records the paper
values as reported metadata, not as a measurement of the current code.

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
damage output: one logit; sigmoid probability; two-class Dice includes both
              undamaged and damaged classes
monitor: val_combined_F1
```

The released configuration stores the training-set inverse-frequency weights
for both tasks, computed as $w_c=N/(C n_c)$ from valid training pixels only.
With the current AOI-aware masking, the values are land-cover
`[1.6963, 2.5613, 1.6376, 0.4630, 2.1080, 0.8263, 0.6390]` and damage
`[0.5054, 46.6519]` (undamaged, damaged).
The configured seed is 42 for this executable reference run; it is not verified
as the seed used by every historical result. The training script uses FP32
(no autocast); mixed-precision use in historical runs has not been verified.
Weighted cross-entropy is summed over valid pixels and divided
by the valid-pixel count; Dice uses epsilon `1e-6`, averages over all classes
(including the undamaged class), and excludes ignore label `255`. For damage,
missing-mask Puerto Rico samples are valid all-zero negatives as described in
the dataset section. At each validation pass, a pooled damage-F1 threshold is
selected on a 0.001 grid; that validation-selected threshold is saved with the
checkpoint and must remain fixed for test evaluation. The test split is never
used for threshold calibration.

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

Historical checkpoint record:

```text
best epoch: 28
best val_combined_F1: 0.8286
historically reported damage threshold: 0.75 (not independently verified)
SHA256: 754a9d0c7a259c37e4b832dacafae2df72b72693f97707a2c3a101c779530ea9
```

The checkpoint is provided through the Zenodo model record above rather than
stored directly in Git. The archived checkpoint predates the single-logit
damage-head correction and is not strictly compatible with the current
manuscript-aligned implementation. The evaluator intentionally uses strict
checkpoint loading and will reject incompatible weights instead of silently
ignoring missing or mismatched parameters. The historical scores and checkpoint
lineage therefore remain distinct from results produced by this code snapshot.

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
