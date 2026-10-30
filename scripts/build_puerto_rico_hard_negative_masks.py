#!/usr/bin/env python3
"""Create explicit zero-damage masks for valid Puerto Rico land pixels.

Released Puerto Rico land masks encode background/outside-AOI as 0 and valid
foreground land-cover classes as 1..7. This script creates a same-size damage
mask with 0 on valid land and 255 outside the annotated AOI, and links each
mask from the corresponding split-manifest row.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


EVENT = "southwest_puerto_rico_2020"
EXPECTED_COUNT = 445


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/lca_eq"))
    args = parser.parse_args()
    root = args.root
    event_root = root / EVENT
    land_dir = event_root / "land_masks"
    damage_dir = event_root / "damage_masks"
    split_dir = root / "splits" / "paper_scene_split"

    land_files = sorted(land_dir.glob("*.png"))
    if len(land_files) != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} land masks, found {len(land_files)}")
    damage_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    target_names: set[str] = set()
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        manifests[split] = (fieldnames, rows)
        for row in rows:
            if row.get("event") != EVENT:
                continue
            land_rel = row["land_mask"]
            land_path = Path(land_rel)
            if land_path.name in target_names:
                raise RuntimeError(f"Duplicate Puerto Rico manifest entry: {land_path.name}")
            target_names.add(land_path.name)

    expected_names = {path.name for path in land_files}
    if target_names != expected_names:
        raise RuntimeError(
            f"Manifest/land-mask mismatch: manifest={len(target_names)}, files={len(expected_names)}"
        )

    generated = 0
    for land_path in land_files:
        land = np.asarray(Image.open(land_path).convert("L"), dtype=np.uint8)
        if not set(np.unique(land).tolist()).issubset(set(range(8))):
            raise RuntimeError(f"Unexpected land-cover IDs in {land_path}")
        damage = np.full(land.shape, 255, dtype=np.uint8)
        damage[land != 0] = 0
        output = damage_dir / land_path.name
        if output.exists():
            existing = np.asarray(Image.open(output).convert("L"), dtype=np.uint8)
            if existing.shape != damage.shape or not np.array_equal(existing, damage):
                raise RuntimeError(f"Existing damage mask differs from generated target: {output}")
        else:
            Image.fromarray(damage).save(output)
            generated += 1

    linked = 0
    for split, (fieldnames, rows) in manifests.items():
        if "damage_mask" not in fieldnames:
            raise RuntimeError(f"Missing damage_mask column in {split}.csv")
        changed = False
        has_event_rows = False
        for row in rows:
            if row.get("event") != EVENT:
                continue
            has_event_rows = True
            expected_rel = f"{EVENT}/damage_masks/{Path(row['land_mask']).name}"
            old = row.get("damage_mask", "").strip()
            if old and old != expected_rel:
                raise RuntimeError(f"Unexpected existing damage path in {split}.csv: {old}")
            if old != expected_rel:
                row["damage_mask"] = expected_rel
                changed = True
            linked += 1
        if changed or has_event_rows:
            atomic_write_csv(split_dir / f"{split}.csv", fieldnames, rows)

    if linked != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} Puerto Rico manifest rows, found {linked}")
    outputs = sorted(damage_dir.glob("*.png"))
    if {path.name for path in outputs} != expected_names:
        raise RuntimeError(f"Unexpected output mask set: expected {EXPECTED_COUNT}, found {len(outputs)}")
    print(f"Puerto Rico hard-negative masks: {len(outputs)} verified; {generated} created; {linked} manifest rows linked.")


if __name__ == "__main__":
    main()
