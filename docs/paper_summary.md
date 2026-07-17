# Paper-Derived Method Summary

## Core Problem

Post-earthquake RGB imagery often contains visual ambiguity. Collapsed buildings and earthquake debris can resemble bare soil, rocky terrain, vegetation shadows, or intact paved areas. A damage detector driven mainly by local appearance may therefore produce false positives on visually similar undamaged surfaces.

## Dataset Formulation

The paper proposes the Land-Cover Anchored Earthquake (LCA-EQ) benchmark with two coupled annotation layers:

- a seven-class land-cover layer
- an independent binary visible-damage layer

LCA-EQ contains 6,683 annotated 1024 x 1024 image patches from eight earthquake events. Damage is represented as a state associated with an underlying land-cover class, rather than as an independent land-cover category.

## Support Classes

Damage-support classes:

- Building
- Road
- Impervious Surface

Non-support context classes:

- Forest
- Farmland
- Water
- Other

## Model

LGDNet contains:

- shared image encoder
- semantic head for land-cover probabilities
- damage head for damage features
- Land-Semantics Gating Module (LSGM)

LSGM aggregates predicted land-cover probabilities into support and non-support probabilities, builds semantic guidance for damage features, and applies soft promotion and suppression to the damage stream.

## Training Objective

The manuscript uses supervised land-cover and damage losses. Damage prediction is the primary target task, while land-cover prediction provides semantic guidance for damage inference.

## Reported Paper Result

Under event-level evaluation against twelve representative segmentation baselines, LGDNet reports:

- Damage IoU: 50.67%
- Damage F1: 67.26%
- Damage IoU improvement over the strongest baseline: 6.91 percentage points

Ablation and error analyses show that land-cover guidance reduces false damage responses on visually similar undamaged surfaces.

This repository provides an implementation scaffold rather than the original full training release.
