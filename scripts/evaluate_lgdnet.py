"""Evaluate LGDNet on an LCA-EQ split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from lgdnet.data import LCAEQDataset
from lgdnet.metrics import confusion_matrix, f1_from_confusion, iou_from_confusion
from lgdnet.models import LGDNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LGDNet on LCA-EQ.")
    parser.add_argument("--config", default="configs/lgdnet_lca_eq.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        cleaned = {k.replace("model.", "", 1): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = config["data"][f"{args.split}_manifest"]
    batch_size = args.batch_size or config["train"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LCAEQDataset(
        manifest,
        root=config["data"].get("root", "."),
        normalize=config["augmentation"].get("normalize", True),
        image_mean=tuple(config["augmentation"].get("mean", [0.485, 0.456, 0.406])),
        image_std=tuple(config["augmentation"].get("std", [0.229, 0.224, 0.225])),
        remap_land_labels=config["data"].get("remap_land_labels", True),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config["train"]["num_workers"])

    model = LGDNet(
        num_land_classes=config["data"]["num_land_classes"],
        decoder_channels=config["model"]["decoder_channels"],
        use_lsgm=config["model"].get("use_lsgm", True),
        host_class_ids=tuple(config["model"].get("host_class_ids", [0, 1, 2])),
        non_host_class_ids=tuple(config["model"].get("non_host_class_ids", [3, 4, 5, 6])),
        pretrained_backbone=False,
    ).to(device)
    load_checkpoint(model, Path(args.checkpoint))
    model.eval()

    land_cm = torch.zeros((config["data"]["num_land_classes"], config["data"]["num_land_classes"]), dtype=torch.long)
    damage_cm = torch.zeros((2, 2), dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            land = batch["land_mask"]
            damage = batch["damage_mask"]
            outputs = model(images)
            land_pred = outputs["land_logits"].argmax(dim=1).cpu()
            damage_prob = torch.softmax(outputs["damage_logits"], dim=1)[:, 1]
            damage_pred = (damage_prob >= config["eval"].get("damage_threshold", 0.5)).long().cpu()
            land_cm += confusion_matrix(land_pred, land, config["data"]["num_land_classes"])
            valid_damage = damage != 255
            if valid_damage.any():
                damage_cm += confusion_matrix(damage_pred[valid_damage], damage[valid_damage], 2)

    land_iou = iou_from_confusion(land_cm)
    land_f1 = f1_from_confusion(land_cm)
    damage_iou = iou_from_confusion(damage_cm)
    damage_f1 = f1_from_confusion(damage_cm)
    combined_f1 = 0.5 * land_f1.mean() + 0.5 * damage_f1[1]

    print(f"split={args.split}")
    print(f"land_mIoU={land_iou.mean().item():.6f}")
    print(f"land_F1={land_f1.mean().item():.6f}")
    print(f"damage_IoU={damage_iou[1].item():.6f}")
    print(f"damage_F1={damage_f1[1].item():.6f}")
    print(f"combined_F1={combined_f1.item():.6f}")


if __name__ == "__main__":
    main()
