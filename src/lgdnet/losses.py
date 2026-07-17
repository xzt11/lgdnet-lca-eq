"""Loss functions for two-layer land-cover and damage supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
    cardinality = (probs + target_one_hot).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """Weighted cross-entropy plus Dice loss."""

    def __init__(self, num_classes: int, class_weights: list[float] | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, target.long(), weight=weights)
        return ce + dice_loss(logits, target, self.num_classes)


class LGDNetLoss(nn.Module):
    """Two-task LGDNet objective used in the manuscript."""

    def __init__(
        self,
        damage_weight: float = 1.0,
        land_weight: float = 0.5,
        damage_class_weights: list[float] | None = None,
        num_land_classes: int = 7,
    ) -> None:
        super().__init__()
        self.damage_weight = damage_weight
        self.land_weight = land_weight
        self.land_loss = CombinedSegmentationLoss(num_land_classes)
        self.damage_loss = CombinedSegmentationLoss(2, damage_class_weights or [1.0, 50.0])

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
