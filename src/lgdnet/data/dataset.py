"""Dataset loader for the cleaned LCA-EQ manifests."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LCAEQDataset(Dataset):
    """Load post-event RGB patches with land-cover and damage masks.

    The CSV manifest must contain:

    - ``image``
    - ``land_mask``
    - ``damage_mask``; may be empty for land-only samples
    - ``event``
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path = ".",
        crop_size: int | None = None,
        random_crop: bool = False,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = Path(root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        with self.manifest.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image = self._read_rgb(row["image"])
        land = self._read_mask(row["land_mask"]).long()

        damage_rel = row.get("damage_mask", "")
        if damage_rel:
            damage = self._read_mask(damage_rel).long()
        else:
            damage = torch.full_like(land, 255)

        if self.crop_size is not None:
            image, land, damage = self._crop(image, land, damage, self.crop_size)

        return {
            "image": image,
            "land_mask": land,
            "damage_mask": damage,
            "event": row.get("event", ""),
        }

    def _resolve(self, rel_path: str) -> Path:
        path = Path(rel_path)
        if path.is_absolute():
            return path
        return self.root / path

    def _read_rgb(self, rel_path: str) -> torch.Tensor:
        arr = np.array(Image.open(self._resolve(rel_path)).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def _read_mask(self, rel_path: str) -> torch.Tensor:
        arr = np.array(Image.open(self._resolve(rel_path)).convert("L"), dtype=np.int64)
        return torch.from_numpy(arr)

    def _crop(
        self,
        image: torch.Tensor,
        land: torch.Tensor,
        damage: torch.Tensor,
        crop_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = land.shape[-2:]
        if height < crop_size or width < crop_size:
            raise ValueError(
                f"crop_size={crop_size} exceeds sample size height={height}, width={width}"
            )
        if self.random_crop:
            top = int(torch.randint(0, height - crop_size + 1, (1,)).item())
            left = int(torch.randint(0, width - crop_size + 1, (1,)).item())
        else:
            top = max((height - crop_size) // 2, 0)
            left = max((width - crop_size) // 2, 0)
        bottom = top + crop_size
        right = left + crop_size
        return image[:, top:bottom, left:right], land[top:bottom, left:right], damage[top:bottom, left:right]
