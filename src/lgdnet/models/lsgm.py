"""Land-Semantics Gating Module for land-conditioned damage features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LandSemanticsGatingModule(nn.Module):
    """Recalibrate damage features with land-cover-derived host priors.

    The module implements the Land-Semantics Gating Module (LSGM) described in
    the paper:

    1. convert land-cover logits to support/non-support probabilities;
    2. pool damage features into semantic prototypes under detached land masks;
    3. estimate host and non-host affinities;
    4. apply bounded residual modulation to the damage feature map.

    The legacy alias ``LocalSemanticGuidanceModule`` is kept for backward
    compatibility with earlier examples.
    """

    def __init__(
        self,
        channels: int,
        host_class_ids: tuple[int, ...] = (0, 1, 2),
        non_host_class_ids: tuple[int, ...] = (3, 4, 5, 6),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.support_class_ids = host_class_ids
        self.non_support_class_ids = non_host_class_ids
        self.eps = eps

        self.damage_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.support_temperature = nn.Parameter(torch.ones(1))
        self.non_support_temperature = nn.Parameter(torch.ones(1))
        self.support_scale = nn.Parameter(torch.zeros(1))
        self.non_support_scale = nn.Parameter(torch.zeros(1))

    def forward(self, damage_features: torch.Tensor, land_logits: torch.Tensor) -> torch.Tensor:
        land_probs = F.softmax(land_logits, dim=1)

        support_mask = self._group_probability(land_probs, self.support_class_ids).detach()
        non_support_mask = self._group_probability(land_probs, self.non_support_class_ids).detach()

        if support_mask.shape[-2:] != damage_features.shape[-2:]:
            support_mask = F.interpolate(support_mask, size=damage_features.shape[-2:], mode="bilinear")
            non_support_mask = F.interpolate(
                non_support_mask,
                size=damage_features.shape[-2:],
                mode="bilinear",
            )

        projected_damage = self.damage_projection(damage_features)
        support_proto = self._masked_average_pool(projected_damage, support_mask)
        non_support_proto = self._masked_average_pool(projected_damage, non_support_mask)

        support_affinity = self._prototype_affinity(projected_damage, support_proto)
        non_support_affinity = self._prototype_affinity(projected_damage, non_support_proto)

        support_gate = torch.sigmoid(support_affinity / self.support_temperature.clamp_min(0.05))
        non_support_gate = torch.sigmoid(
            non_support_affinity / self.non_support_temperature.clamp_min(0.05)
        )
        lambda_support = torch.sigmoid(self.support_scale)
        lambda_non_support = torch.sigmoid(self.non_support_scale)
        modulation = 1.0 + lambda_support * support_gate - lambda_non_support * non_support_gate
        return damage_features * modulation.clamp_min(self.eps)

    def _group_probability(self, probabilities: torch.Tensor, class_ids: tuple[int, ...]) -> torch.Tensor:
        valid_ids = [class_id for class_id in class_ids if 0 <= class_id < probabilities.shape[1]]
        if not valid_ids:
            shape = (probabilities.shape[0], 1, probabilities.shape[2], probabilities.shape[3])
            return probabilities.new_zeros(shape)
        return probabilities[:, valid_ids].sum(dim=1, keepdim=True)

    def _prototype_affinity(self, key: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        key = F.normalize(key, p=2, dim=1, eps=self.eps)
        prototype = F.normalize(prototype, p=2, dim=1, eps=self.eps)
        return (key * prototype).sum(dim=1, keepdim=True)

    def _masked_average_pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = (features * mask).sum(dim=(2, 3), keepdim=True)
        denominator = mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return numerator / denominator


LocalSemanticGuidanceModule = LandSemanticsGatingModule
