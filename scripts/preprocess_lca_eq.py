"""Validate and summarize an LCA-EQ release directory.

The full image archive is distributed outside Git. This script checks that the
restored local directory follows the paper layout and that the split manifests
point to existing image, land-mask, and damage-mask files where applicable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_SPLITS = {"train": 5490, "val": 353, "test": 840}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an LCA-EQ release directory.")
    parser.add_argument("--root", default="data/lca_eq", help="Path to LCA-EQ root directory.")
    parser.add_argument("--require-images", action="store_true", help="Fail if RGB patches are missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    split_dir = root / "splits" / "paper_scene_split"
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    for split, expected_count in EXPECTED_SPLITS.items():
        manifest = split_dir / f"{split}.csv"
        rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
        if len(rows) != expected_count:
            raise RuntimeError(f"{manifest}: expected {expected_count}, got {len(rows)}")

        missing_images = missing_land = missing_damage = 0
        for row in rows:
            if not (root / row["image"]).exists():
                missing_images += 1
            if not (root / row["land_mask"]).exists():
                missing_land += 1
            damage_mask = row.get("damage_mask", "")
            if damage_mask and not (root / damage_mask).exists():
                missing_damage += 1

        if missing_land or missing_damage:
            raise RuntimeError(
                f"{split}: missing_land={missing_land}, missing_damage={missing_damage}"
            )
        if args.require_images and missing_images:
            raise RuntimeError(f"{split}: missing_images={missing_images}")

        print(
            f"{split}: rows={len(rows)} missing_images={missing_images} "
            f"missing_land={missing_land} missing_damage={missing_damage}"
        )

    print("LCA-EQ validation finished.")


if __name__ == "__main__":
    main()
