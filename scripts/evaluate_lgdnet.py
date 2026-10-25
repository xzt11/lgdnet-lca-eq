"""Evaluate LGDNet; test thresholds are calibrated on validation data only."""

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
    parser = argparse.ArgumentParser(description="Evaluate LGDNet on an LCA-EQ split.")
    parser.add_argument("--config", default="configs/lgdnet_lca_eq.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit threshold. If omitted for test, it is selected on validation by pooled F1.",
    )
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")

    cleaned = {}
    for name, value in state.items():
        while name.startswith("module.") or name.startswith("model."):
            name = name.split(".", 1)[1]
        cleaned[name] = value
    # Strict loading prevents silently evaluating checkpoints with a different
    # architecture or damage-head channel count.
    model.load_state_dict(cleaned, strict=True)


def make_dataset(config: dict, split: str) -> LCAEQDataset:
    aug = config["augmentation"]
    return LCAEQDataset(
        config["data"][f"{split}_manifest"],
        root=config["data"].get("root", "."),
        normalize=aug.get("normalize", True),
        image_mean=tuple(aug.get("mean", [0.485, 0.456, 0.406])),
        image_std=tuple(aug.get("std", [0.229, 0.224, 0.225])),
        remap_land_labels=config["data"].get("remap_land_labels", True),
        hard_negative_events=tuple(config["data"].get("hard_negative_events", [])),
    )


@torch.no_grad()
def collect_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_land_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return LC confusion matrix and 0.001-binned damage score histograms."""

    land_cm = torch.zeros((num_land_classes, num_land_classes), dtype=torch.long)
    positive_hist = torch.zeros(1001, dtype=torch.long, device=device)
    negative_hist = torch.zeros(1001, dtype=torch.long, device=device)
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        land = batch["land_mask"]
        damage = batch["damage_mask"].to(device, non_blocking=True)
        outputs = model(images)
        land_pred = outputs["land_logits"].argmax(dim=1).cpu()
        land_cm += confusion_matrix(land_pred, land, num_land_classes)

        damage_prob = torch.softmax(outputs["damage_logits"], dim=1)[:, 1]
        valid = damage != 255
        if valid.any():
            valid_target = damage[valid]
            if torch.any((valid_target < 0) | (valid_target > 1)):
                raise ValueError("Damage GT must be binary 0/1 or ignore=255")
            score_bins = (damage_prob[valid] * 1000).floor().long().clamp_(0, 1000)
            target_device = valid_target
            positive_hist += torch.bincount(
                score_bins[target_device == 1], minlength=1001
            )
            negative_hist += torch.bincount(
                score_bins[target_device == 0], minlength=1001
            )
    return land_cm, positive_hist.cpu(), negative_hist.cpu()


def select_f1_threshold(positive_hist: torch.Tensor, negative_hist: torch.Tensor) -> float:
    tp = positive_hist.flip(0).cumsum(0).flip(0).double()
    fp = negative_hist.flip(0).cumsum(0).flip(0).double()
    fn = positive_hist.sum().double() - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return int(torch.argmax(f1).item()) / 1000.0


def damage_confusion(
    positive_hist: torch.Tensor,
    negative_hist: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Damage threshold must be in [0,1], got {threshold}")
    threshold_id = int(round(threshold * 1000))
    tp = positive_hist[threshold_id:].sum()
    fp = negative_hist[threshold_id:].sum()
    fn = positive_hist.sum() - tp
    tn = negative_hist.sum() - fp
    return torch.tensor([[tn, fp], [fn, tp]], dtype=torch.long)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = LGDNet(
        num_land_classes=config["data"]["num_land_classes"],
        decoder_channels=model_cfg["decoder_channels"],
        use_lsgm=model_cfg.get("use_lsgm", True),
        host_class_ids=tuple(model_cfg.get("host_class_ids", [0, 1, 2])),
        non_host_class_ids=tuple(model_cfg.get("non_host_class_ids", [3, 4, 5, 6])),
        pretrained_backbone=False,
        transformer_layers=model_cfg.get("transformer_layers", 2),
        transformer_heads=model_cfg.get("transformer_heads", 8),
        transformer_dropout=model_cfg.get("transformer_dropout", 0.1),
        mamba_state=model_cfg.get("mamba_state", 16),
        mamba_conv=model_cfg.get("mamba_conv", 4),
        mamba_expansion=model_cfg.get("mamba_expansion", 2),
        lsgm_embedding_channels=model_cfg.get("lsgm_embedding_channels"),
        lsgm_rho=model_cfg.get("lsgm_rho", 0.9),
        lsgm_eps=model_cfg.get("lsgm_eps", 1e-6),
    ).to(device)
    load_checkpoint(model, Path(args.checkpoint))

    batch_size = args.batch_size or config["train"].get("val_batch_size", 1)
    workers = config["train"].get("num_workers", 4)
    num_classes = config["data"]["num_land_classes"]

    if args.threshold is not None:
        selected_threshold = args.threshold
    elif args.split == "test":
        val_loader = DataLoader(
            make_dataset(config, "val"), batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=torch.cuda.is_available()
        )
        _, val_positive, val_negative = collect_split(model, val_loader, device, num_classes)
        selected_threshold = select_f1_threshold(val_positive, val_negative)
        print(f"threshold_source=validation_pooled_f1 threshold={selected_threshold:.3f}")
    elif args.split == "val":
        # Validation-only calibration is useful for reporting the selected
        # operating point; the resulting metrics are not test metrics.
        val_loader = DataLoader(
            make_dataset(config, "val"), batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=torch.cuda.is_available()
        )
        _, val_positive, val_negative = collect_split(model, val_loader, device, num_classes)
        selected_threshold = select_f1_threshold(val_positive, val_negative)
        print(f"threshold_source=validation_pooled_f1 threshold={selected_threshold:.3f}")
    else:
        raise ValueError("For --split train, provide --threshold explicitly; do not calibrate on train data.")

    loader = DataLoader(
        make_dataset(config, args.split), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available()
    )
    land_cm, positive_hist, negative_hist = collect_split(model, loader, device, num_classes)
    damage_cm = damage_confusion(positive_hist, negative_hist, selected_threshold)
    land_iou = iou_from_confusion(land_cm)
    land_f1 = f1_from_confusion(land_cm)
    damage_iou = iou_from_confusion(damage_cm)
    damage_f1 = f1_from_confusion(damage_cm)
    combined_f1 = 0.5 * land_f1.mean() + 0.5 * damage_f1[1]

    print(f"split={args.split}")
    print(f"damage_threshold={selected_threshold:.3f}")
    print(f"land_mIoU={land_iou.mean().item():.6f}")
    print(f"land_F1={land_f1.mean().item():.6f}")
    print(f"damage_IoU={damage_iou[1].item():.6f}")
    print(f"damage_F1={damage_f1[1].item():.6f}")
    print(f"combined_F1={combined_f1.item():.6f}")


if __name__ == "__main__":
    main()
