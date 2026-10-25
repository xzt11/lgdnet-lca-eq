"""Two-task LGDNet losses for land-cover and damage supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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
    if torch.any((target[valid] < 0) | (target[valid] >= num_classes)):
        raise ValueError(f"Target contains labels outside [0, {num_classes - 1}] and ignore={ignore_index}")

    if logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected logits [B,{num_classes},H,W], got {tuple(logits.shape)}"
        )
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
        dice_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_eps = dice_eps
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.any(target != self.ignore_index):
            return logits.sum() * 0.0
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        valid = target != self.ignore_index
        if torch.any((target[valid] < 0) | (target[valid] >= self.num_classes)):
            raise ValueError(
                f"Target contains labels outside [0, {self.num_classes - 1}] "
                f"and ignore={self.ignore_index}"
            )
        # The manuscript defines the weighted CE as a sum divided by the
        # number of valid pixels (rather than PyTorch's weighted-mean divisor).
        ce_map = F.cross_entropy(
            logits, target.long(), weight=weights, ignore_index=self.ignore_index, reduction="none"
        )
        ce = ce_map[valid].sum() / valid.sum().clamp_min(1)
        return ce + dice_loss(
            logits, target, self.num_classes, ignore_index=self.ignore_index, eps=self.dice_eps
        )


class BinaryDamageLoss(nn.Module):
    """Weighted two-class cross-entropy + Dice for the damage logits."""

    def __init__(
        self,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
        dice_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_eps = dice_eps
        weights = class_weights or [1.0, 1.0]
        if len(weights) != 2:
            raise ValueError("Damage class_weights must contain [undamaged, damaged] weights")
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError(f"Expected damage logits [B,2,H,W], got {tuple(logits.shape)}")
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        if torch.any((target[valid] < 0) | (target[valid] > 1)):
            raise ValueError("Damage targets must be binary 0/1 or the configured ignore label")

        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        ce_map = F.cross_entropy(
            logits,
            target.long(),
            weight=weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        ce = ce_map[valid].sum() / valid.sum().clamp_min(1)
        dice = dice_loss(
            logits, target, num_classes=2, ignore_index=self.ignore_index, eps=self.dice_eps
        )
        return ce + dice


class LGDNetLoss(nn.Module):
    """Paper two-task objective: land-cover loss plus damage loss."""

    def __init__(
        self,
        damage_weight: float = 1.0,
        land_weight: float = 0.5,
        damage_class_weights: list[float] | None = None,
        land_class_weights: list[float] | None = None,
        num_land_classes: int = 7,
        dice_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.damage_weight = damage_weight
        self.land_weight = land_weight
        self.land_loss = CombinedSegmentationLoss(
            num_land_classes,
            land_class_weights or [1.0] * num_land_classes,
            dice_eps=dice_eps,
        )
        self.damage_loss = BinaryDamageLoss(damage_class_weights, dice_eps=dice_eps)

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
