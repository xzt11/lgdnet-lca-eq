"""Manifest-based loader for LCA-EQ-style two-layer annotations."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class LCAEQDataset(Dataset):
    """Load post-event images with land-cover and damage masks from a CSV manifest.

    The manifest must contain the columns ``image``, ``land_mask``, ``damage_mask``
    and ``event``. Paths are interpreted relative to ``root`` unless absolute.
    """

    def __init__(self, manifest: str | Path, root: str | Path = ".", image_size: int | None = 384) -> None:
        self.root = Path(root)
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.root / self.manifest
        self.image_size = image_size
        with self.manifest.open("r", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        required = {"image", "land_mask", "damage_mask", "event"}
        if self.rows and not required.issubset(self.rows[0]):
            missing = sorted(required - set(self.rows[0]))
            raise ValueError(f"Manifest is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        image = Image.open(self._resolve(row["image"])).convert("RGB")
        land = Image.open(self._resolve(row["land_mask"])).convert("L")
        damage = Image.open(self._resolve(row["damage_mask"])).convert("L")

        image_tensor = TF.to_tensor(image)
        land_tensor = torch.as_tensor(list(land.getdata()), dtype=torch.long).reshape(land.height, land.width)
        damage_tensor = torch.as_tensor(list(damage.getdata()), dtype=torch.long).reshape(damage.height, damage.width)

        if self.image_size is not None and image_tensor.shape[-1] != self.image_size:
            size = (self.image_size, self.image_size)
            image_tensor = F.interpolate(image_tensor[None], size=size, mode="bilinear", align_corners=False)[0]
            land_tensor = F.interpolate(land_tensor[None, None].float(), size=size, mode="nearest")[0, 0].long()
            damage_tensor = F.interpolate(damage_tensor[None, None].float(), size=size, mode="nearest")[0, 0].long()

        return {
            "image": image_tensor,
            "land_mask": land_tensor,
            "damage_mask": damage_tensor,
            "event": row["event"],
        }
