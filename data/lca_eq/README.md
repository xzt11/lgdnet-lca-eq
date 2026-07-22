# LCA-EQ Dataset

This directory contains the cleaned 8-event LCA-EQ release assets used by the
LGDNet paper.

Directory names follow the event names reported in the manuscript table.

Each event directory follows this layout:

- `images/`: expected location of post-event RGB image patches reconstructed by users with lawful access to the source imagery
- `land_masks/`: land-cover labels
- `damage_masks/`: binary visible-damage labels when available

Split manifests are intentionally limited to the event-level scene split used in
the manuscript:

- `splits/paper_scene_split/train.csv`
- `splits/paper_scene_split/val.csv`
- `splits/paper_scene_split/test.csv`
- `events.csv`: event-level metadata following the paper table

Original Google Earth image pixels are not redistributed in this GitHub release.
The split CSV files keep the expected relative image paths so that users can
place reconstructed 1024 x 1024 RGB patches under the corresponding `images/`
directories before training or evaluation.

Note: `southwest_puerto_rico_2020` currently has land-cover masks but no damage
masks in this release. No synthetic all-zero damage masks were created.
