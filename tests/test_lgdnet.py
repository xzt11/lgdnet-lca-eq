import torch

from lgdnet.losses import LGDNetLoss
from lgdnet.models import LGDNet, LandSemanticsGatingModule


def test_lgdnet_forward_shapes():
    model = LGDNet(num_land_classes=7, decoder_channels=32)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["land_logits"].shape == (2, 7, 64, 64)
    assert outputs["damage_logits"].shape == (2, 2, 64, 64)
    assert outputs["aux_damage_logits"].shape == (2, 2, 64, 64)


def test_lsgm_preserves_feature_shape():
    module = LandSemanticsGatingModule(channels=16)
    features = torch.randn(2, 16, 32, 32)
    land_logits = torch.randn(2, 7, 32, 32)
    gated = module(features, land_logits)
    assert gated.shape == features.shape


def test_composite_loss_runs():
    model = LGDNet(num_land_classes=7, decoder_channels=32)
    criterion = LGDNetLoss()
    outputs = model(torch.randn(2, 3, 64, 64))
    land = torch.randint(0, 7, (2, 64, 64))
    damage = torch.randint(0, 2, (2, 64, 64))
    losses = criterion(outputs, land, damage)
    assert losses["total"].ndim == 0
