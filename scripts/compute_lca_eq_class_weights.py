"""Compute LCA-EQ inverse-frequency class weights from a split manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


PAPER_LAND_LABEL_MAPPING = {
    0: 255,
    1: 1,
    2: 4,
    3: 0,
    4: 3,
    5: 2,
    6: 6,
    7: 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LCA-EQ class weights.")
    parser.add_argument("--root", default="data/lca_eq")
    parser.add_argument(
        "--manifest",
        default="data/lca_eq/splits/paper_scene_split/train.csv",
    )
    parser.add_argument(
        "--hard-negative-event",
        action="append",
        default=["southwest_puerto_rico_2020"],
        help="Event whose missing damage masks should count as all-zero negatives.",
    )
    return parser.parse_args()


def inverse_frequency_weights(counts: np.ndarray) -> np.ndarray:
    return counts.sum() / (len(counts) * np.maximum(counts, 1))


def remap_land(mask: np.ndarray) -> np.ndarray:
    remapped = np.full(mask.shape, 255, dtype=np.uint8)
    for src, dst in PAPER_LAND_LABEL_MAPPING.items():
        remapped[mask == src] = dst
    return remapped


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest)
    hard_negative_events = set(args.hard_negative_event or [])

    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    land_counts = np.zeros(7, dtype=np.int64)
    damage_counts = np.zeros(2, dtype=np.int64)

    for row in rows:
        land = np.array(Image.open(root / row["land_mask"]).convert("L"), dtype=np.uint8)
        land = remap_land(land)
        for class_id in range(7):
            land_counts[class_id] += np.count_nonzero(land == class_id)

        damage_rel = row.get("damage_mask", "").strip()
        if damage_rel:
            damage = np.array(Image.open(root / damage_rel).convert("L"), dtype=np.uint8)
            damage_counts[0] += np.count_nonzero(damage == 0)
            damage_counts[1] += np.count_nonzero(damage == 1)
        elif row.get("event", "") in hard_negative_events:
            damage_counts[0] += land.size

    land_weights = inverse_frequency_weights(land_counts)
    damage_weights = inverse_frequency_weights(damage_counts)

    print("land_counts:", land_counts.tolist())
    print("damage_counts:", damage_counts.tolist())
    print("land_class_weights:", [round(float(x), 4) for x in land_weights])
    print("damage_class_weights:", [round(float(x), 4) for x in damage_weights])


if __name__ == "__main__":
    main()
