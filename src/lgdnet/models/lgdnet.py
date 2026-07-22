"""LGDNet implementation with land-conditioned damage routing."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from lgdnet.models.lsgm import LocalSemanticGuidanceModule


class SpatialRefinementBlock(nn.Module):
    """Two-layer convolutional refinement used by encoder and task heads."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(x)


class PostEventEncoder(nn.Module):
    """Shared post-event visual encoder.

    LGDNet intentionally uses a single post-event RGB stream. The encoder returns
    multi-scale feature maps that are later fused before branching into the land
    and damage tasks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = SpatialRefinementBlock(3, 64)
        self.stage2 = SpatialRefinementBlock(64, 128, stride=2)
        self.stage3 = SpatialRefinementBlock(128, 256, stride=2)
        self.stage4 = SpatialRefinementBlock(256, 512, stride=2)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(image)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4


class LandConditionedDecoder(nn.Module):
    """Top-down fusion decoder shared by land-cover and damage heads."""

    def __init__(self, decoder_channels: int) -> None:
        super().__init__()
        self.project_stage4 = nn.Conv2d(512, decoder_channels, kernel_size=1)
        self.project_stage3 = nn.Conv2d(256, decoder_channels, kernel_size=1)
        self.project_stage2 = nn.Conv2d(128, decoder_channels, kernel_size=1)
        self.project_stage1 = nn.Conv2d(64, decoder_channels, kernel_size=1)
        self.scene_fusion = SpatialRefinementBlock(decoder_channels, decoder_channels)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        top = self.project_stage4(f4)
        mid_high = self.project_stage3(f3) + F.interpolate(top, size=f3.shape[-2:], mode="bilinear")
        mid_low = self.project_stage2(f2) + F.interpolate(mid_high, size=f2.shape[-2:], mode="bilinear")
        full = self.project_stage1(f1) + F.interpolate(mid_low, size=f1.shape[-2:], mode="bilinear")
        return self.scene_fusion(full)


class TaskPredictionHeads(nn.Module):
    """Land-cover prediction and damage evidence extraction heads."""

    def __init__(self, decoder_channels: int, num_land_classes: int) -> None:
        super().__init__()
        self.land_classifier = nn.Conv2d(decoder_channels, num_land_classes, kernel_size=1)
        self.damage_evidence = SpatialRefinementBlock(decoder_channels, decoder_channels)
        self.damage_classifier = nn.Conv2d(decoder_channels, 2, kernel_size=1)

    def land_logits(self, shared_features: torch.Tensor) -> torch.Tensor:
        return self.land_classifier(shared_features)

    def damage_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        return self.damage_evidence(shared_features)

    def damage_logits(self, calibrated_damage_features: torch.Tensor) -> torch.Tensor:
        return self.damage_classifier(calibrated_damage_features)


class LGDNet(nn.Module):
    """Dual-head segmentation network with land-semantics gating.

    The network follows the paper interface: one post-event RGB input and two
    supervised outputs, land-cover segmentation and binary damage segmentation.
    """

    def __init__(
        self,
        num_land_classes: int = 8,
        decoder_channels: int = 128,
        use_lsgm: bool = True,
        host_class_ids: tuple[int, ...] = (1, 3, 5),
        non_host_class_ids: tuple[int, ...] = (2, 4, 6, 7),
    ) -> None:
        super().__init__()
        self.use_lsgm = use_lsgm

        self.post_event_encoder = PostEventEncoder()
        self.scene_decoder = LandConditionedDecoder(decoder_channels)
        self.task_heads = TaskPredictionHeads(decoder_channels, num_land_classes)
        self.local_semantic_guidance = LocalSemanticGuidanceModule(
            decoder_channels,
            host_class_ids=host_class_ids,
            non_host_class_ids=non_host_class_ids,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]

        encoded_features = self.post_event_encoder(image)
        shared_scene_features = self.scene_decoder(encoded_features)

        land_logits = self.task_heads.land_logits(shared_scene_features)
        damage_features = self.task_heads.damage_features(shared_scene_features)
        if self.use_lsgm:
            damage_features = self.local_semantic_guidance(damage_features, land_logits)

        damage_logits = self.task_heads.damage_logits(damage_features)

        return {
            "land_logits": F.interpolate(land_logits, size=input_size, mode="bilinear"),
            "damage_logits": F.interpolate(damage_logits, size=input_size, mode="bilinear"),
        }
