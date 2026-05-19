# LCA-EQ-Style Data Card

This repository expects data following the two-layer formulation described in the paper.

## Inputs

- post-event VHR RGB image patches
- optional pre-event imagery for future semantic-prior extensions

## Labels

Each image has two masks:

- land-cover mask with seven classes
- binary damage-state mask

## Recommended Split

Use event-aware splits whenever possible to measure spatial and cross-event generalization. The paper reports a 6:2:2 train/validation/test split by event samples.

## Important Notes

- Do not treat damage as a standalone land-cover class.
- Keep land-cover and damage-state masks aligned pixel-by-pixel.
- Preserve non-host classes because they are central to measuring commission errors.
- Document imagery source, license, event date, post-event acquisition timing, and spatial resolution before redistribution.
