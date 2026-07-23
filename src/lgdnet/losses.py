"""Two-task LGDNet losses for land-cover and damage supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_LAND_CLASS_WEIGHTS = [1.6963, 2.5613, 1.6376, 0.4630, 2.1080, 0.8263, 0.6390]
DEFAULT_DAMAGE_CLASS_WEIGHTS = [0.5049, 51.2585]


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = target != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0

    probs = F.softmax(logits, dim=1)
    safe_target = target.masked_fill(~valid, 0)
    target_one_hot = F.one_hot(safe_target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probs = probs * valid
    target_one_hot = target_one_hot * valid
    intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
    cardinality = (probs + target_one_hot).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """Weighted cross-entropy plus Dice loss."""

    def __init__(
        self,
        num_classes: int,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.any(target != self.ignore_index):
            return logits.sum() * 0.0
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, target.long(), weight=weights, ignore_index=self.ignore_index)
        return ce + dice_loss(logits, target, self.num_classes, ignore_index=self.ignore_index)


class LGDNetLoss(nn.Module):
    """Paper two-task objective: land-cover loss plus damage loss."""

    def __init__(
        self,
        damage_weight: float = 1.0,
        land_weight: float = 0.5,
        damage_class_weights: list[float] | None = None,
        land_class_weights: list[float] | None = None,
        num_land_classes: int = 7,
    ) -> None:
        super().__init__()
        self.damage_weight = damage_weight
        self.land_weight = land_weight
        self.land_loss = CombinedSegmentationLoss(
            num_land_classes,
            land_class_weights or DEFAULT_LAND_CLASS_WEIGHTS,
        )
        self.damage_loss = CombinedSegmentationLoss(
            2,
            damage_class_weights or DEFAULT_DAMAGE_CLASS_WEIGHTS,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        land_mask: torch.Tensor,
        damage_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        land = self.land_loss(outputs["land_logits"], land_mask)
        damage = self.damage_loss(outputs["damage_logits"], damage_mask)
        total = self.damage_weight * damage + self.land_weight * land
        return {"total": total, "land": land, "damage": damage}
