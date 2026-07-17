"""Land-Semantics Gating Module."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LandSemanticsGatingModule(nn.Module):
    """Modulate damage features using land-cover-derived semantic guidance."""

    def __init__(
        self,
        channels: int,
        support_class_ids: tuple[int, ...] = (0, 1, 2),
        non_support_class_ids: tuple[int, ...] = (3, 4, 5, 6),
        host_class_ids: tuple[int, ...] | None = None,
        non_host_class_ids: tuple[int, ...] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.support_class_ids = host_class_ids if host_class_ids is not None else support_class_ids
        self.non_support_class_ids = (
            non_host_class_ids if non_host_class_ids is not None else non_support_class_ids
        )
        self.eps = eps

        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.support_temperature = nn.Parameter(torch.ones(1))
        self.non_support_temperature = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, damage_features: torch.Tensor, land_logits: torch.Tensor) -> torch.Tensor:
        land_probs = F.softmax(land_logits, dim=1)

        support_mask = land_probs[:, self.support_class_ids].sum(dim=1, keepdim=True).detach()
        non_support_mask = land_probs[:, self.non_support_class_ids].sum(dim=1, keepdim=True).detach()

        if support_mask.shape[-2:] != damage_features.shape[-2:]:
            support_mask = F.interpolate(support_mask, size=damage_features.shape[-2:], mode="bilinear")
            non_support_mask = F.interpolate(
                non_support_mask,
                size=damage_features.shape[-2:],
                mode="bilinear",
            )

        support_proto = self._masked_average_pool(damage_features, support_mask)
        non_support_proto = self._masked_average_pool(damage_features, non_support_mask)

        key = self.key(damage_features)
        value = self.value(damage_features)

        support_similarity = (key * support_proto).sum(dim=1, keepdim=True)
        non_support_similarity = (key * non_support_proto).sum(dim=1, keepdim=True)

        suppress_gate = 1.0 - torch.sigmoid(
            non_support_similarity / self.non_support_temperature.clamp_min(0.05)
        )
        promote_gate = torch.sigmoid(
            support_similarity / self.support_temperature.clamp_min(0.05)
        )

        return damage_features * suppress_gate + self.alpha * (promote_gate * value)

    def _masked_average_pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = (features * mask).sum(dim=(2, 3), keepdim=True)
        denominator = mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return numerator / denominator
