"""Compact LGDNet implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from lgdnet.models.lsgm import LandSemanticsGatingModule


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LGDNet(nn.Module):
    """Dual-head segmentation network with land-semantics gating."""

    def __init__(
        self,
        num_land_classes: int = 7,
        decoder_channels: int = 128,
        use_lsgm: bool = True,
        support_class_ids: tuple[int, ...] = (0, 1, 2),
        non_support_class_ids: tuple[int, ...] = (3, 4, 5, 6),
        host_class_ids: tuple[int, ...] | None = None,
        non_host_class_ids: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.use_lsgm = use_lsgm
        if host_class_ids is not None:
            support_class_ids = host_class_ids
        if non_host_class_ids is not None:
            non_support_class_ids = non_host_class_ids

        self.stem = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128, stride=2)
        self.enc3 = ConvBlock(128, 256, stride=2)
        self.enc4 = ConvBlock(256, 512, stride=2)

        self.lateral4 = nn.Conv2d(512, decoder_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, decoder_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, decoder_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(64, decoder_channels, kernel_size=1)

        self.fuse = ConvBlock(decoder_channels, decoder_channels)
        self.semantic_head = nn.Conv2d(decoder_channels, num_land_classes, kernel_size=1)
        self.damage_feature_head = ConvBlock(decoder_channels, decoder_channels)

        self.lsgm = LandSemanticsGatingModule(
            decoder_channels,
            support_class_ids=support_class_ids,
            non_support_class_ids=non_support_class_ids,
        )
        self.damage_head = nn.Conv2d(decoder_channels, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]

        e1 = self.stem(image)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        p4 = self.lateral4(e4)
        p3 = self.lateral3(e3) + F.interpolate(p4, size=e3.shape[-2:], mode="bilinear")
        p2 = self.lateral2(e2) + F.interpolate(p3, size=e2.shape[-2:], mode="bilinear")
        p1 = self.lateral1(e1) + F.interpolate(p2, size=e1.shape[-2:], mode="bilinear")
        features = self.fuse(p1)

        land_logits = self.semantic_head(features)
        damage_features = self.damage_feature_head(features)

        if self.use_lsgm:
            damage_features = self.lsgm(damage_features, land_logits)

        damage_logits = self.damage_head(damage_features)

        return {
            "land_logits": F.interpolate(land_logits, size=input_size, mode="bilinear"),
            "damage_logits": F.interpolate(damage_logits, size=input_size, mode="bilinear"),
        }
