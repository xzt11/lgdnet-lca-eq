import torch
from PIL import Image

from lgdnet.data import LCAEQDataset
from lgdnet.losses import LGDNetLoss
from lgdnet.models import LGDNet, LandSemanticsGatingModule


def test_lgdnet_forward_shapes():
    model = LGDNet(num_land_classes=7, decoder_channels=32, pretrained_backbone=False)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["land_logits"].shape == (2, 7, 64, 64)
    assert outputs["damage_logits"].shape == (2, 2, 64, 64)
    assert set(outputs) == {"land_logits", "damage_logits"}


def test_lsgm_preserves_feature_shape():
    module = LandSemanticsGatingModule(channels=16)
    features = torch.randn(2, 16, 32, 32)
    land_logits = torch.randn(2, 7, 32, 32)
    gated = module(features, land_logits)
    assert gated.shape == features.shape


def test_two_task_loss_runs():
    model = LGDNet(num_land_classes=7, decoder_channels=32, pretrained_backbone=False)
    criterion = LGDNetLoss()
    outputs = model(torch.randn(2, 3, 64, 64))
    land = torch.randint(0, 7, (2, 64, 64))
    damage = torch.randint(0, 2, (2, 64, 64))
    losses = criterion(outputs, land, damage)
    assert losses["total"].ndim == 0
    assert set(losses) == {"total", "land", "damage"}


def test_damage_ignore_mask_runs_for_land_only_samples():
    model = LGDNet(num_land_classes=7, decoder_channels=32, pretrained_backbone=False)
    criterion = LGDNetLoss()
    outputs = model(torch.randn(2, 3, 64, 64))
    land = torch.randint(0, 7, (2, 64, 64))
    damage = torch.full((2, 64, 64), 255)
    losses = criterion(outputs, land, damage)
    assert losses["damage"].ndim == 0
    assert torch.isfinite(losses["total"])


def test_missing_puerto_rico_damage_mask_is_hard_negative(tmp_path):
    event = "southwest_puerto_rico_2020"
    event_dir = tmp_path / event
    image_dir = event_dir / "images"
    land_dir = event_dir / "land_masks"
    image_dir.mkdir(parents=True)
    land_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image_dir / "sample.png")
    Image.new("L", (8, 8), color=3).save(land_dir / "sample.png")
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "image,land_mask,damage_mask,event\n"
        f"{event}/images/sample.png,{event}/land_masks/sample.png,,{event}\n",
        encoding="utf-8",
    )

    dataset = LCAEQDataset(
        manifest,
        root=tmp_path,
        hard_negative_events=(event,),
        normalize=False,
    )
    sample = dataset[0]
    assert torch.equal(sample["damage_mask"], torch.zeros((8, 8), dtype=torch.long))
