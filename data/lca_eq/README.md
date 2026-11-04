# LCA-EQ Dataset

This directory contains the paper-split metadata and land-cover/damage labels
for the original eight-event LCA-EQ release used for the paper. Source labels
are retained without post-hoc support-class filtering.

Directory names follow the event names reported in the manuscript table.

Each event directory follows this layout:

- `images/`: expected location of post-event RGB image patches reconstructed by users with lawful access to the source imagery
- `land_masks/`: land-cover labels. The training loader remaps released masks
  to the seven foreground classes used in the paper and converts background or
  outside-AOI pixels to ignore label `255`.
- `damage_masks/`: binary visible-damage labels when available

Split manifests are intentionally limited to the event-level scene split used in
the manuscript:

- `splits/paper_scene_split/train.csv`
- `splits/paper_scene_split/val.csv`
- `splits/paper_scene_split/test.csv`
- `events.csv`: event-level metadata following the paper table

## Damage labels and source licensing

Damage masks use `0` for no visible damage, `1` for visible damage, and `255`
for ignore where applicable. They are preserved as supplied; preprocessing does
not rewrite them according to land-cover class. Pixels outside the annotated AOI
are ignored by the loader using the land-cover mask.

Original Google Earth image pixels are not redistributed in this GitHub release.
The split CSV files keep the expected relative image paths so that users can
place reconstructed 1024 x 1024 RGB patches under the corresponding `images/`
directories before training or evaluation.

`southwest_puerto_rico_2020` contains 445 explicit damage masks and no positive
visible-damage labels. Each mask is `0` on valid land-cover pixels (raw IDs
1..7) and `255` where the land-cover mask is background/outside the annotated
AOI (raw land-cover ID 0). The train manifest points to these files, so the
patches contribute directly as valid all-zero hard negatives.

See [`LICENSE_DATA.md`](LICENSE_DATA.md). The root repository MIT license
covers source code, not satellite imagery. Google Earth RGB pixels are not
redistributed here. Maxar imagery is subject to CC BY-NC 4.0 only where the
specific source asset is covered by the Maxar Open Data Program license. The
versioned Zenodo v1.1 release retains the RGB archive, adds the 445 Puerto Rico
hard-negative masks, and documents Google Earth imagery under the separate
permission obtained for that public release: [Zenodo v1.1](https://doi.org/10.5281/zenodo.23934838).
