# LCA-EQ Release Structure

This directory records the public release structure for the LCA-EQ benchmark.

The original Google Earth historical image tiles are not redistributed. The release is designed around derived research assets that support reproducibility for users with lawful access to the corresponding source imagery.

## Included in Git

- `metadata/events.csv`: event-level metadata and patch counts
- `splits/event_splits.csv`: event-level train/validation/test assignment
- `splits/*_events.txt`: event lists for each subset
- `splits/*_manifest.csv`: manifest headers for image, land-cover mask, damage mask, and event fields
- directory-level READMEs for AOI footprints, crop grids, and labels

## Derived Assets

The manuscript data availability statement refers to the following derived assets:

- land-cover labels
- visible-damage annotations
- AOI footprints
- acquisition metadata
- crop grids and patch indices
- event-level split files
- preprocessing and evaluation scripts
- model code and configuration files

Large binary label archives, AOI files, and crop-grid files can be distributed as repository files, GitHub Release assets, or an external archive linked from this directory. The original source imagery remains subject to the licence terms of the imagery provider.
