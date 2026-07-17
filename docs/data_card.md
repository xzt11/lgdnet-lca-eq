# LCA-EQ Data Card

LCA-EQ follows a two-layer annotation design for land-cover-conditioned post-earthquake damage mapping. Visible damage is represented as a state associated with an underlying land-cover class, not as a mutually exclusive land-cover category.

## Inputs

- post-event VHR RGB image patches
- source patch size: 1024 x 1024 pixels
- random training crop size: 384 x 384 pixels
- approximate ground sampling distance: 0.3--0.5 m

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

## Event-Level Split

LCA-EQ uses an event-level split. All patches from the same earthquake event are assigned to the same subset.

- Train: Noto Peninsula 2024, Turkiye-Syria 2023, Mandalay 2025, Southwest Puerto Rico 2020
- Validation: Yangbi 2021, Nippes 2021
- Test: Al Haouz 2023, Sulawesi 2018

The selected Southwest Puerto Rico AOIs contain no positive visible-damage labels and are included in the training set to provide diverse undamaged land-cover context and hard negative samples.

## Data Availability and Licensing

The original Google Earth historical imagery used to construct LCA-EQ is not redistributed because its reuse is governed by third-party imagery licence terms. The repository provides or indexes derived research assets needed for reproducible use, including land-cover labels, visible-damage annotations, AOI footprints, acquisition metadata, crop grids, split files, scripts, model code, and configuration files.

Exact reconstruction of the original image pixels depends on lawful access to the corresponding historical imagery.

## Important Notes

- Do not treat damage as a land-cover class.
- Keep land-cover and damage masks aligned pixel-by-pixel.
- Preserve non-support classes because they are central to measuring false damage responses.
- Document imagery source, licence terms, event date, post-event acquisition timing, spatial resolution, and patch extraction settings.
