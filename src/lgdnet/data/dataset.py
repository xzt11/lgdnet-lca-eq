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


def inverse_frequency_class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Return the paper's ``N / (C * n_c)`` weights from valid pixel counts."""
    if counts.ndim != 1 or torch.any(counts <= 0):
        raise ValueError("Every class must have a positive valid-pixel count")
    counts = counts.to(torch.float64)
    return (counts.sum() / (counts.numel() * counts)).to(torch.float32)


def compute_training_class_weights(
    dataset: "LCAEQDataset", num_land_classes: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count valid remapped labels over the unaugmented training manifest."""
    land_counts = torch.zeros(num_land_classes, dtype=torch.int64)
    damage_counts = torch.zeros(2, dtype=torch.int64)
    for row in dataset.rows:
        land = torch.from_numpy(
            np.array(Image.open(dataset._resolve(row["land_mask"])).convert("L"), dtype=np.int64)
        )
        if dataset.remap_land_labels:
            land = dataset._remap_land_mask(land)
        valid_land = land != 255
        for class_id in range(num_land_classes):
            land_counts[class_id] += (land == class_id).sum()

        damage_path = row.get("damage_mask", "").strip()
        if damage_path:
            damage = torch.from_numpy(
                np.array(Image.open(dataset._resolve(damage_path)).convert("L"), dtype=np.int64)
            )
            if damage.shape != land.shape:
                raise ValueError(f"Land/damage mask shape mismatch for {row['image']}")
            valid_damage = valid_land & ((damage == 0) | (damage == 1))
            for class_id in range(2):
                damage_counts[class_id] += ((damage == class_id) & valid_damage).sum()
        elif row.get("event", "") in dataset.hard_negative_events:
            damage_counts[0] += valid_land.sum()

    return inverse_frequency_class_weights(land_counts), inverse_frequency_class_weights(damage_counts)


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
        hard_negative_events: tuple[str, ...] = (),
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
        self.hard_negative_events = set(hard_negative_events)
        with self.manifest.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image = self._read_rgb(row["image"])
        land = self._read_mask(row["land_mask"]).long()
        if image.shape[-2:] != land.shape[-2:]:
            raise ValueError(
                f"Image/land-mask shape mismatch for {row['image']}: "
                f"{tuple(image.shape[-2:])} vs {tuple(land.shape[-2:])}"
            )
        if self.remap_land_labels:
            land = self._remap_land_mask(land)

        damage_rel = row.get("damage_mask", "")
        if damage_rel:
            damage = self._read_mask(damage_rel).long()
            if damage.shape != land.shape:
                raise ValueError(
                    f"Land/damage-mask shape mismatch for {row['image']}: "
                    f"{tuple(land.shape)} vs {tuple(damage.shape)}"
                )
            # AOI-exterior/background pixels are ignored for both tasks,
            # regardless of how the released binary damage PNG encodes them.
            damage = damage.masked_fill(land == 255, 255)
        elif row.get("event", "") in self.hard_negative_events:
            damage = torch.zeros_like(land)
            damage = damage.masked_fill(land == 255, 255)
        else:
            damage = torch.full_like(land, 255)

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
