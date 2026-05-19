# Paper-Derived Method Summary

## Core Problem

Post-earthquake RGB imagery often contains visual ambiguity: collapsed buildings and earthquake debris can resemble bare soil, dry riverbeds, rocky terrain, vegetation shadows, or intact paved areas. A texture-driven damage detector may therefore produce false positives outside plausible structural hosts.

## Dataset Formulation

The paper proposes the LCA-EQ benchmark with two coupled annotation layers:

- land-cover layer with seven classes
- binary damage-state layer

Damage is represented as a state associated with land-cover hosts, not as an independent land-cover category.

## Host Classes

Plausible anthropogenic host classes:

- Buildings
- Roads
- Impervious Surfaces

Non-host context classes:

- Forest
- Farmland
- Water
- Others

## Model

LGDNet contains:

- shared image encoder
- semantic head for land-cover probabilities
- damage head for damage features
- Land-Semantics Gating Module (LSGM)

LSGM aggregates predicted land-cover probabilities into host and non-host masks, stops gradients from damage loss into the semantic branch, builds host/non-host damage-feature prototypes, and applies soft suppression/promotion gates to the damage stream.

## Training Objective

The reported objective is:

```text
L_total = 1.0 * L_damage + 0.5 * L_land + 0.3 * L_aux
```

Both land-cover and damage losses combine cross-entropy and Dice supervision. Damage loss uses strong positive weighting because damaged pixels are highly sparse.

## Reported Paper Result

On the LCA-EQ test set, the paper reports:

- Damage IoU: 50.67%
- Damage F1: 66.69%
- reduced Damage Commission Error compared with dual-head baseline without LSGM

This repository provides an implementation scaffold rather than the original full training release.
