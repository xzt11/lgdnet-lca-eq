# C10: Architecture and training reproducibility

This document is aligned to the revised manuscript and response package dated
2026-10-25. In particular, the damage head is **two logits** (undamaged and
damaged), normalized with a two-class Softmax. It is not the one-logit/sigmoid
formulation found in an earlier code snapshot.

## Architecture implemented by this repository

Input is a single post-event RGB image. For an input of size `H × W`, the shared
ImageNet-initialized ResNet-101 (`3/4/23/3` bottleneck blocks) returns:

| Feature | Channels | Resolution |
| --- | ---: | ---: |
| C2 | 256 | `H/4 × W/4` |
| C3 | 512 | `H/8 × W/8` |
| C4 | 1024 | `H/16 × W/16` |
| C5 | 2048 | `H/32 × W/32` |

The stride-32 C5 map is projected to 448 channels and flattened into spatial
tokens. A two-layer Transformer encoder uses 8 attention heads, a 1792-wide
feed-forward layer (4× embedding width), GELU, pre-LayerNorm, and dropout 0.1.
The output is reshaped to the stride-32 feature map.

The FPN decoder uses 1×1 lateral projections from C4, C3, and C2 to 448
channels. Each top-down feature is bilinearly upsampled and added to its lateral
feature, then refined by one Mamba decoder block, in order at strides 16, 8, and
4. Each block uses `d_state=16`, `d_conv=4`, expansion 2, pre-LayerNorm and a
learnable residual scale initialized to zero. Two 3×3 Conv-BN-SiLU layers
refine the final stride-4 shared feature. For 384×384 crops, the FPN levels are
24×24, 48×48, and 96×96; for 1024×1024 inference patches they are 64×64,
128×128, and 256×256.

Two task-specific branches begin from the shared stride-4 feature. The
land-cover branch applies 3×3 Conv-BN-SiLU, Dropout2d(0.1), and a 1×1
seven-class classifier. The damage-feature branch applies 3×3 Conv-BN-SiLU and
Dropout2d(0.1). Its feature is modulated by LSGM using the upsampled semantic
logits; the 448-channel damage feature is projected inside LSGM to the
configured 224-dimensional embedding. The final damage classifier is a 1×1
convolution with **two output channels**. After upsampling, Softmax channel 1 is
the damaged-class probability. The binary decision is `p(damaged) >= t`;
argmax is equivalent only at `t=0.5`.

## Executable training recipe in the repository

The default configuration is `configs/lgdnet_lca_eq.yaml`; it specifies the
paper split, 384×384 random crops, 30 epochs, batch size 8, AdamW with Lookahead
(`k=5`, `alpha=0.5`), pretrained-encoder learning rate `6e-5`, newly initialized
module learning rate `6e-4`, weight decay `0.01`, and cosine annealing. Training
augmentation applies scale jittering in `[0.75, 1.25]`, independent horizontal
and vertical flips, and ImageNet per-channel normalization. Crop origins are
uniformly sampled over each jittered image; no positive-centered crop sampler
is used. Padding, when required, is ignored by assigning label 255.

The total loss is `1.0 * L_damage + 0.5 * L_land`. Each task loss is weighted
cross-entropy plus Dice. Damage Dice averages both classes (undamaged and
damaged); land-cover Dice averages all seven classes. Label 255 is excluded.
The repository recipe calculates inverse-frequency weights from valid pixels
in the training manifest as `w_c = N / (C * n_c)` and writes the resulting
weights into the checkpoint. This makes the released recipe data-deterministic;
it does **not** establish that those are the unavailable exact weights from the
historical reported run. Dice smoothing uses `1e-6` in this executable recipe;
the historical numerical smoothing constant was not recovered.

The reference script seeds Python, NumPy, PyTorch, CUDA, and DataLoader workers
with seed 42. It runs in FP32 (no autocast or GradScaler). The configured
checkpoint monitor is `val_combined_F1`, defined as the unweighted mean of
seven-class land-cover macro-F1 and positive-class damage F1. At each validation
pass, damage threshold selection scans pooled validation probabilities on a
0.001 grid and maximizes positive-class damage F1. The best epoch is the epoch
with the highest configured combined validation score; test labels are not
used to select the threshold. The numerical best epoch and threshold of the
historical result are not recoverable from the materials supplied with this
release.

Run from the repository root after restoring the dataset at the paths in the
manifest:

```bash
pip install -e ".[dev]"
python scripts/train.py --config configs/lgdnet_lca_eq.yaml
python scripts/evaluate_lgdnet.py \
  --config configs/lgdnet_lca_eq.yaml \
  --checkpoint outputs/lgdnet_lca_eq/best.ckpt \
  --split test
```

Evaluation without `--threshold` selects it on the validation split and then
keeps it fixed while scoring the requested split. For strict test reporting,
record that selected value and pass it explicitly on later evaluation runs.
Checkpoints store the epoch metrics, selected validation threshold, and
computed class weights.

## Historical facts versus unavailable run records

The 2026-10-25 manuscript reports PyTorch 2.7.1, 30 epochs, batch size 8,
384×384 crops, the optimizer and learning rates above, augmentation, full
1024×1024 inference, validation combined-F1 checkpoint selection, and a
validation-selected threshold fixed for test evaluation. The associated C10
response explicitly says that the following historical run details were not
available: exact numerical class weights, numerical smoothing constant,
random seed, Lookahead hyperparameters, selected epoch, numerical selected
threshold, and mixed-precision setting. The code defaults above are therefore
an executable reference recipe where the historical record is missing, not
claims that those values were used in the reported run.

The repository currently provides an executable LGDNet training and evaluation
configuration. It does not contain recovered, result-specific executable
configurations for every baseline, ablation, table, or figure in the manuscript.
Those should be added only from actual run records; this document does not
invent baseline settings or label them reproducible.

The Zenodo checkpoint's compatibility with this two-logit implementation has
not been verified by a strict load in this source checkout. Historical
checkpoint metadata in the README is retained as archival metadata only and
must not be confused with a reproduced run.
