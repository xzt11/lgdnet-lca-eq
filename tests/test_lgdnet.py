import torch

from lgdnet.losses import LGDNetLoss
from lgdnet.models import LGDNet, LandSemanticsGatingModule


def test_lgdnet_forward_shapes():
    model = LGDNet(num_land_classes=8, decoder_channels=32)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["land_logits"].shape == (2, 8, 64, 64)
    assert outputs["damage_logits"].shape == (2, 2, 64, 64)
    assert set(outputs) == {"land_logits", "damage_logits"}


def test_lsgm_preserves_feature_shape():
    module = LandSemanticsGatingModule(channels=16)
    features = torch.randn(2, 16, 32, 32)
    land_logits = torch.randn(2, 8, 32, 32)
    gated = module(features, land_logits)
    assert gated.shape == features.shape


def test_two_task_loss_runs():
    model = LGDNet(num_land_classes=8, decoder_channels=32)
    criterion = LGDNetLoss()
    outputs = model(torch.randn(2, 3, 64, 64))
    land = torch.randint(0, 8, (2, 64, 64))
    damage = torch.randint(0, 2, (2, 64, 64))
    losses = criterion(outputs, land, damage)
    assert losses["total"].ndim == 0
    assert set(losses) == {"total", "land", "damage"}


def test_damage_ignore_mask_runs_for_land_only_samples():
    model = LGDNet(num_land_classes=8, decoder_channels=32)
    criterion = LGDNetLoss()
    outputs = model(torch.randn(2, 3, 64, 64))
    land = torch.randint(0, 8, (2, 64, 64))
    damage = torch.full((2, 64, 64), 255)
    losses = criterion(outputs, land, damage)
    assert losses["damage"].ndim == 0
    assert torch.isfinite(losses["total"])
