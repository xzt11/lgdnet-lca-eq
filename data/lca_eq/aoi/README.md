# LCA-EQ AOI Metadata and Footprint Candidates

This directory contains event-level AOI metadata and a GeoJSON of source-image
footprint candidates:

- `event_aoi_metadata.csv` uses the canonical `event_id` values from
  `events.csv` and the split manifests.
- `LCA_EQ_AOI_Footprints.geojson` contains 12 polygon features derived from
  georeferenced source-raster extents or non-transparent raster footprints.

## Important status and limitations

The GeoJSON geometries are **candidate source footprints, not verified final
annotation/AOI boundaries**. The feature properties preserve this status.
They have not been checked against the final label masks or the exact patch
selection boundary, so they must not be presented as definitive annotation
boundaries without further verification. The Yangbi feature is explicitly
extent-only, and the Sulawesi feature's event year is marked unverified.

GeoJSON feature `event_id` values are source labels and do not all match the
canonical IDs in `events.csv`. Their intended correspondences are:

| GeoJSON source label | Canonical `events.csv` event |
| --- | --- |
| `turkey_2023` | `turkiye_syria_2023` |
| `noto_japan_2024` | `noto_peninsula_japan_2024` |
| `morocco_2023` | `al_haouz_morocco_2023` |
| `myanmar_2025` | `mandalay_myanmar_2025` |
| `yunnan_2021` | `yangbi_china_2021` |
| `puerto_rico_2020` | `southwest_puerto_rico_2020` |
| `haiti_2021` | `nippes_haiti_2021` |
| `sulawesi_event_unverified` | `sulawesi_indonesia_2018` (event identity/year requires verification) |

The `source_crs` property records the source raster CRS; GeoJSON geometry
coordinates are longitude/latitude. Consult each feature's `aoi_status`,
`boundary_basis`, and source metadata before using it. Redistribution rights
for the original imagery remain separate from the release of these footprint
polygons.
