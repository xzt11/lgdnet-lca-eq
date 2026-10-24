import torch
from PIL import Image

from lgdnet.data import LCAEQDataset
from lgdnet.losses import LGDNetLoss
from lgdnet.metrics import confusion_matrix
from lgdnet.models import LGDNet, LandSemanticsGatingModule


def test_lgdnet_forward_shapes():
    model = LGDNet(num_land_classes=7, decoder_channels=32, pretrained_backbone=False)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["land_logits"].shape == (2, 7, 64, 64)
    assert outputs["damage_logits"].shape == (2, 1, 64, 64)
    assert set(outputs) == {"land_logits", "damage_logits"}


def test_lsgm_preserves_feature_shape():
    module = LandSemanticsGatingModule(channels=16)
    assert module.damage_projection.out_channels == 8
    features = torch.randn(2, 16, 32, 32)
    land_logits = torch.randn(2, 7, 32, 32)
    gated = module(features, land_logits)
    assert gated.shape == features.shape


def test_lsgm_uses_paper_bounded_modulation_and_backpropagates_semantics():
    module = LandSemanticsGatingModule(channels=8, rho=0.9)
    features = torch.randn(2, 8, 6, 6, requires_grad=True)
    land_logits = torch.randn(2, 7, 6, 6, requires_grad=True)
    output = module(features, land_logits)
    output.square().mean().backward()
    assert land_logits.grad is not None
    assert torch.isfinite(land_logits.grad).all()
    assert torch.allclose(
        torch.nn.functional.softplus(module.support_temperature_raw) + module.eps,
        torch.ones(1),
        atol=1e-5,
    )
    assert 0.0 < (module.rho * torch.sigmoid(module.non_support_scale)).item() < 0.9


def test_two_task_loss_runs():
    model = LGDNet(num_land_classes=7, decoder_channels=32, pretrained_backbone=False)
    criterion = LGDNetLoss()
    outputs = model(torch.randn(2, 3, 64, 64))
    land = torch.randint(0, 7, (2, 64, 64))
    damage = torch.randint(0, 2, (2, 64, 64))
    losses = criterion(outputs, land, damage)
    assert losses["total"].ndim == 0
    assert set(losses) == {"total", "land", "damage"}


def test_single_logit_damage_loss_includes_both_dice_classes_and_ignores_255():
    model = LGDNet(num_land_classes=7, decoder_channels=32, pretrained_backbone=False)
    criterion = LGDNetLoss()
    outputs = model(torch.randn(1, 3, 64, 64))
    assert outputs["damage_logits"].shape[1] == 1
    land = torch.zeros((1, 64, 64), dtype=torch.long)
    damage = torch.zeros((1, 64, 64), dtype=torch.long)
    damage[:, :8, :8] = 1
    damage[:, 8:12, :12] = 255
    losses = criterion(outputs, land, damage)
    assert torch.isfinite(losses["damage"])
    losses["total"].backward()
    assert model.heads.damage_classifier.weight.grad is not None


def test_ignore_255_never_enters_confusion_matrix():
    target = torch.tensor([0, 1, 255, 7, -1])
    pred = torch.tensor([0, 0, 1, 1, 0])
    cm = confusion_matrix(pred, target, num_classes=2)
    assert cm.tolist() == [[1, 0], [1, 0]]
    assert cm.sum().item() == 2


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


def test_damage_pixels_outside_aoi_are_ignored(tmp_path):
    event = "sample_event"
    event_dir = tmp_path / event
    image_dir = event_dir / "images"
    land_dir = event_dir / "land_masks"
    damage_dir = event_dir / "damage_masks"
    for directory in (image_dir, land_dir, damage_dir):
        directory.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image_dir / "sample.png")
    # Raw label 0 is background/outside AOI; raw label 3 maps to Building.
    land = torch.full((8, 8), 3, dtype=torch.uint8)
    land[:2, :] = 0
    Image.fromarray(land.numpy()).save(land_dir / "sample.png")
    damage = torch.ones((8, 8), dtype=torch.uint8)
    Image.fromarray(damage.numpy()).save(damage_dir / "sample.png")
    manifest = tmp_path / "sample.csv"
    manifest.write_text(
        "image,land_mask,damage_mask,event\n"
        f"{event}/images/sample.png,{event}/land_masks/sample.png,"
        f"{event}/damage_masks/sample.png,{event}\n",
        encoding="utf-8",
    )
    sample = LCAEQDataset(manifest, root=tmp_path, normalize=False)[0]
    assert torch.all(sample["damage_mask"][:2] == 255)
    assert torch.all(sample["damage_mask"][2:] == 1)
