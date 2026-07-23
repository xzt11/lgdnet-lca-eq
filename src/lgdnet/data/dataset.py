"""Dataset loader for LCA-EQ paper manifests."""

from __future__ import annotations

from pathlib import Path
import csv
import random

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


PAPER_LAND_LABEL_MAPPING = {
    0: 255,  # background / outside annotated AOIs
    1: 1,  # Roads
    2: 4,  # Farmland
    3: 0,  # Buildings
    4: 3,  # Forest
    5: 2,  # Impervious surface
    6: 6,  # Other
    7: 5,  # Water
}


class LCAEQDataset(Dataset):
    """Load post-event RGB patches, seven-class land labels, and damage masks.

    The released masks may store background as label ``0`` and foreground land
    categories as ``1..7``. For the paper setting, background/outside-AOI pixels
    are remapped to ``255`` and ignored by loss/evaluation, while foreground
    categories are remapped to the seven-class paper order:

    ``Building, Road, Impervious Surface, Forest, Farmland, Water, Other``.
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path = ".",
        crop_size: int | None = None,
        random_crop: bool = False,
        scale_jitter: tuple[float, float] | None = None,
        hflip: bool = False,
        vflip: bool = False,
        normalize: bool = True,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        remap_land_labels: bool = True,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = Path(root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.scale_jitter = scale_jitter
        self.hflip = hflip
        self.vflip = vflip
        self.normalize = normalize
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.remap_land_labels = remap_land_labels
        with self.manifest.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image = self._read_rgb(row["image"])
        land = self._read_mask(row["land_mask"]).long()
        if self.remap_land_labels:
            land = self._remap_land_mask(land)

        damage_rel = row.get("damage_mask", "")
        damage = self._read_mask(damage_rel).long() if damage_rel else torch.full_like(land, 255)

        if self.scale_jitter is not None:
            image, land, damage = self._scale_jitter(image, land, damage, self.scale_jitter)
        if self.crop_size is not None:
            image, land, damage = self._crop(image, land, damage, self.crop_size)
        if self.hflip and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            land = torch.flip(land, dims=[1])
            damage = torch.flip(damage, dims=[1])
        if self.vflip and random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            land = torch.flip(land, dims=[0])
            damage = torch.flip(damage, dims=[0])
        if self.normalize:
            image = (image - self.image_mean) / self.image_std

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

    def _remap_land_mask(self, mask: torch.Tensor) -> torch.Tensor:
        remapped = torch.full_like(mask, 255)
        for src, dst in PAPER_LAND_LABEL_MAPPING.items():
            remapped[mask == src] = dst
        return remapped

    def _scale_jitter(
        self,
        image: torch.Tensor,
        land: torch.Tensor,
        damage: torch.Tensor,
        scale_range: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = random.uniform(*scale_range)
        height, width = land.shape
        new_size = (max(1, round(height * scale)), max(1, round(width * scale)))
        image = F.interpolate(
            image.unsqueeze(0), size=new_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        land = F.interpolate(land[None, None].float(), size=new_size, mode="nearest").squeeze(0).squeeze(0).long()
        damage = (
            F.interpolate(damage[None, None].float(), size=new_size, mode="nearest")
            .squeeze(0)
            .squeeze(0)
            .long()
        )
        return image, land, damage

    def _crop(
        self,
        image: torch.Tensor,
        land: torch.Tensor,
        damage: torch.Tensor,
        crop_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = land.shape[-2:]
        pad_h = max(crop_size - height, 0)
        pad_w = max(crop_size - width, 0)
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
            land = F.pad(land, (0, pad_w, 0, pad_h), value=255)
            damage = F.pad(damage, (0, pad_w, 0, pad_h), value=255)
            height, width = land.shape[-2:]

        if self.random_crop:
            top = int(torch.randint(0, height - crop_size + 1, (1,)).item())
            left = int(torch.randint(0, width - crop_size + 1, (1,)).item())
        else:
            top = max((height - crop_size) // 2, 0)
            left = max((width - crop_size) // 2, 0)
        bottom = top + crop_size
        right = left + crop_size
        return image[:, top:bottom, left:right], land[top:bottom, left:right], damage[top:bottom, left:right]
