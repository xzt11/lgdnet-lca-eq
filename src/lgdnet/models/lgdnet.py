"""LGDNet architecture described in the LCA-EQ paper.

The public implementation follows the manuscript-level model definition:

- a single post-event RGB input;
- an ImageNet-pretrained ResNet-101 encoder;
- a Transformer bottleneck on the coarsest feature map;
- an FPN-style top-down decoder with Mamba-style spatial sequence blocks;
- separate land-cover and damage heads;
- Land-Semantics Gating Module (LSGM) routing damage features with land-cover
  probabilities.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet101_Weights, resnet101

from lgdnet.models.lsgm import LandSemanticsGatingModule


class ConvBNAct(nn.Module):
    """3x3 convolution, batch normalization, and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet101Encoder(nn.Module):
    """ImageNet-pretrained ResNet-101 feature encoder."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet101(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = (256, 512, 1024, 2048)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(image)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class TransformerBottleneck(nn.Module):
    """Transformer encoder applied to the stride-32 feature map."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        del channels
        x = self.proj(x)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(self.encoder(tokens))
        return tokens.transpose(1, 2).reshape(bsz, -1, height, width)


class MambaDecoderBlock(nn.Module):
    """Mamba-style spatial sequence mixing block for decoder feature maps.

    The block keeps the repository self-contained while preserving the paper
    design choice: local convolutional context is combined with gated
    sequence-wise mixing in the decoder. It can be replaced by an external
    ``mamba_ssm`` block without changing the LGDNet interface.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden)
        self.row_mixer = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden)
        self.col_mixer = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        value, gate = self.in_proj(x).chunk(2, dim=1)
        value = self.depthwise(value)

        bsz, channels, height, width = value.shape
        row_tokens = value.permute(0, 2, 1, 3).reshape(bsz * height, channels, width)
        row_tokens = self.row_mixer(row_tokens).reshape(bsz, height, channels, width)
        row_tokens = row_tokens.permute(0, 2, 1, 3)

        col_tokens = value.permute(0, 3, 1, 2).reshape(bsz * width, channels, height)
        col_tokens = self.col_mixer(col_tokens).reshape(bsz, width, channels, height)
        col_tokens = col_tokens.permute(0, 2, 3, 1)

        mixed = F.silu(row_tokens + col_tokens) * torch.sigmoid(gate)
        return residual + self.gamma * self.out_proj(mixed)


class FPNMambaDecoder(nn.Module):
    """FPN decoder with Mamba-style refinement after each top-down fusion."""

    def __init__(self, encoder_channels: tuple[int, int, int, int], decoder_channels: int) -> None:
        super().__init__()
        c2, c3, c4, c5 = encoder_channels
        self.bottleneck = TransformerBottleneck(c5, decoder_channels)
        self.lat4 = nn.Conv2d(c4, decoder_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, decoder_channels, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, decoder_channels, kernel_size=1)
        self.refine4 = MambaDecoderBlock(decoder_channels)
        self.refine3 = MambaDecoderBlock(decoder_channels)
        self.refine2 = MambaDecoderBlock(decoder_channels)
        self.out = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels),
            ConvBNAct(decoder_channels, decoder_channels),
        )

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = features
        p5 = self.bottleneck(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.refine4(p4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.refine3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.refine2(p2)
        return self.out(p2)


class TaskHeads(nn.Module):
    """Separate land-cover and damage prediction branches."""

    def __init__(self, channels: int, num_land_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.land_branch = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, num_land_classes, kernel_size=1),
        )
        self.damage_branch = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Dropout2d(dropout),
        )
        self.damage_classifier = nn.Conv2d(channels, 2, kernel_size=1)

    def forward_land(self, features: torch.Tensor) -> torch.Tensor:
        return self.land_branch(features)

    def forward_damage_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.damage_branch(features)

    def forward_damage_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.damage_classifier(features)


class LGDNet(nn.Module):
    """Paper LGDNet: ResNet-101 + Transformer + FPN/Mamba decoder + LSGM."""

    def __init__(
        self,
        num_land_classes: int = 7,
        decoder_channels: int = 256,
        use_lsgm: bool = True,
        host_class_ids: tuple[int, ...] = (0, 1, 2),
        non_host_class_ids: tuple[int, ...] = (3, 4, 5, 6),
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.use_lsgm = use_lsgm
        self.encoder = ResNet101Encoder(pretrained=pretrained_backbone)
        self.decoder = FPNMambaDecoder(self.encoder.out_channels, decoder_channels)
        self.heads = TaskHeads(decoder_channels, num_land_classes)
        self.land_semantics_gating = LandSemanticsGatingModule(
            decoder_channels,
            host_class_ids=host_class_ids,
            non_host_class_ids=non_host_class_ids,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        features = self.encoder(image)
        shared_features = self.decoder(features)

        land_logits_low = self.heads.forward_land(shared_features)
        damage_features = self.heads.forward_damage_features(shared_features)
        if self.use_lsgm:
            damage_features = self.land_semantics_gating(damage_features, land_logits_low)
        damage_logits_low = self.heads.forward_damage_logits(damage_features)

        return {
            "land_logits": F.interpolate(
                land_logits_low, size=input_size, mode="bilinear", align_corners=False
            ),
            "damage_logits": F.interpolate(
                damage_logits_low, size=input_size, mode="bilinear", align_corners=False
            ),
        }
