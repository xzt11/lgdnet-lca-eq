"""Land-Semantics Gating Module."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LandSemanticsGatingModule(nn.Module):
    """Modulate damage features with land-cover-derived host priors.

    The module follows the paper's core idea: land-cover probabilities are grouped
    into host and non-host masks, detached from the damage loss path, and used to
    build feature prototypes that suppress non-host-like damage responses and
    promote host-like responses.
    """

    def __init__(
        self,
        channels: int,
        host_class_ids: tuple[int, ...] = (0, 1, 2),
        non_host_class_ids: tuple[int, ...] = (3, 4, 5, 6),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.host_class_ids = host_class_ids
        self.non_host_class_ids = non_host_class_ids
        self.eps = eps

        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.host_temperature = nn.Parameter(torch.ones(1))
        self.non_host_temperature = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, damage_features: torch.Tensor, land_logits: torch.Tensor) -> torch.Tensor:
        land_probs = F.softmax(land_logits, dim=1)

        host_mask = land_probs[:, self.host_class_ids].sum(dim=1, keepdim=True).detach()
        non_host_mask = land_probs[:, self.non_host_class_ids].sum(dim=1, keepdim=True).detach()

        if host_mask.shape[-2:] != damage_features.shape[-2:]:
            host_mask = F.interpolate(host_mask, size=damage_features.shape[-2:], mode="bilinear")
            non_host_mask = F.interpolate(
                non_host_mask,
                size=damage_features.shape[-2:],
                mode="bilinear",
            )

        host_proto = self._masked_average_pool(damage_features, host_mask)
        non_host_proto = self._masked_average_pool(damage_features, non_host_mask)

        key = self.key(damage_features)
        value = self.value(damage_features)

        host_similarity = (key * host_proto).sum(dim=1, keepdim=True)
        non_host_similarity = (key * non_host_proto).sum(dim=1, keepdim=True)

        suppress_gate = 1.0 - torch.sigmoid(non_host_similarity / self.non_host_temperature.clamp_min(0.05))
        promote_gate = torch.sigmoid(host_similarity / self.host_temperature.clamp_min(0.05))

        return damage_features * suppress_gate + self.alpha * (promote_gate * value)

    def _masked_average_pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = (features * mask).sum(dim=(2, 3), keepdim=True)
        denominator = mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return numerator / denominator
