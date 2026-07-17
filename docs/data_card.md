# LCA-EQ-Style Data Card

This repository expects data following the two-layer formulation described in the paper. Visible damage is represented as a state associated with an underlying land-cover class, not as a mutually exclusive land-cover category.

## Inputs

- post-event VHR RGB image patches
- approximate ground sampling distance: 0.3--0.5 m
- patch size used by LCA-EQ: 1024 x 1024 pixels
- optional pre-event imagery may be used only for future semantic-prior extensions

## Labels

Each image has two spatially aligned masks:

- land-cover mask with seven mutually exclusive classes
- independent binary visible-damage mask

Land-cover classes:

| Id | Class |
| --- | --- |
| 0 | Building |
| 1 | Road |
| 2 | Impervious Surface |
| 3 | Forest |
| 4 | Farmland |
| 5 | Water |
| 6 | Other |

## Damage Scope

The benchmark focuses on visible built-environment damage in post-event RGB VHR imagery. Building, Road, and Impervious Surface are treated as damage-support classes. Forest, Farmland, Water, and Other provide non-support context and hard negative samples under the defined task scope.

Natural-terrain impacts such as landslides, liquefaction, ground rupture, and slope failures are outside the present label scope unless they are visually associated with mapped built-environment damage.

## Recommended Split

Use event-level splits whenever possible to measure cross-event generalization. All patches from the same earthquake event should be assigned to the same subset.

## Data Availability and Licensing

The original Google Earth historical imagery used to construct LCA-EQ is not redistributed because its reuse is governed by third-party imagery licence terms. To support reproducibility, release or retain derived research assets such as land-cover labels, visible-damage annotations, AOI footprints, acquisition-time metadata, patch indices, split files, preprocessing scripts, and evaluation scripts. Exact reconstruction of the original image pixels depends on lawful access to the corresponding historical imagery.

## Important Notes

- Do not treat damage as a land-cover class.
- Keep land-cover and damage masks aligned pixel-by-pixel.
- Preserve non-support classes because they are central to measuring false damage responses.
- Document imagery source, licence terms, event date, post-event acquisition timing, spatial resolution, and patch extraction settings.
