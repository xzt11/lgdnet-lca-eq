"""Train LGDNet with the paper LCA-EQ setting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch.utils.data import DataLoader

from lgdnet.data import LCAEQDataset
from lgdnet.losses import LGDNetLoss
from lgdnet.metrics import confusion_matrix, f1_from_confusion, iou_from_confusion
from lgdnet.models import LGDNet


class Lookahead:
    """Small Lookahead wrapper for AdamW.

    The base optimizer keeps the parameter groups and learning rates; this
    wrapper only performs the periodic slow-weight interpolation.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.steps = 0
        self.slow_weights = [
            param.detach().clone()
            for group in optimizer.param_groups
            for param in group["params"]
            if param.requires_grad
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.k != 0:
            return
        idx = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                slow = self.slow_weights[idx]
                slow.add_(param.data - slow, alpha=self.alpha)
                param.data.copy_(slow)
                idx += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LGDNet on LCA-EQ.")
    parser.add_argument("--config", default="configs/lgdnet_lca_eq.yaml")
    return parser.parse_args()


def make_dataset(config: dict, split: str, train: bool) -> LCAEQDataset:
    aug = config["augmentation"]
    return LCAEQDataset(
        config["data"][f"{split}_manifest"],
        root=config["data"].get("root", "."),
        crop_size=config["data"].get("image_size", 384),
        random_crop=train,
        scale_jitter=tuple(aug["scale_jitter"]) if train and aug.get("scale_jitter") else None,
        hflip=train and aug.get("horizontal_flip", False),
        vflip=train and aug.get("vertical_flip", False),
        normalize=aug.get("normalize", True),
        image_mean=tuple(aug.get("mean", [0.485, 0.456, 0.406])),
        image_std=tuple(aug.get("std", [0.229, 0.224, 0.225])),
        remap_land_labels=config["data"].get("remap_land_labels", True),
    )


def parameter_groups(model: LGDNet, config: dict) -> list[dict]:
    pretrained_lr = config["optimizer"]["pretrained_lr"]
    new_lr = config["optimizer"]["new_lr"]
    weight_decay = config["optimizer"]["weight_decay"]
    backbone_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            backbone_params.append(param)
        else:
            new_params.append(param)
    return [
        {"params": backbone_params, "lr": pretrained_lr, "weight_decay": weight_decay},
        {"params": new_params, "lr": new_lr, "weight_decay": weight_decay},
    ]


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["image"].to(device, non_blocking=True),
        batch["land_mask"].to(device, non_blocking=True),
        batch["damage_mask"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(
    model: LGDNet,
    loader: Iterable,
    criterion: LGDNetLoss,
    device: torch.device,
    num_land_classes: int,
    damage_threshold: float,
) -> dict[str, float]:
    model.eval()
    land_cm = torch.zeros((num_land_classes, num_land_classes), dtype=torch.long)
    damage_cm = torch.zeros((2, 2), dtype=torch.long)
    total_loss = 0.0
    batches = 0

    for batch in loader:
        images, land, damage = move_batch(batch, device)
        outputs = model(images)
        losses = criterion(outputs, land, damage)
        total_loss += float(losses["total"].detach())
        batches += 1

        land_pred = outputs["land_logits"].argmax(dim=1).cpu()
        damage_prob = torch.softmax(outputs["damage_logits"], dim=1)[:, 1]
        damage_pred = (damage_prob >= damage_threshold).long().cpu()
        land_cpu = land.cpu()
        damage_cpu = damage.cpu()
        land_cm += confusion_matrix(land_pred, land_cpu, num_land_classes)
        valid_damage = damage_cpu != 255
        if valid_damage.any():
            damage_cm += confusion_matrix(damage_pred[valid_damage], damage_cpu[valid_damage], 2)

    land_iou = iou_from_confusion(land_cm)
    land_f1 = f1_from_confusion(land_cm)
    damage_iou = iou_from_confusion(damage_cm)
    damage_f1 = f1_from_confusion(damage_cm)
    combined_f1 = 0.5 * land_f1.mean() + 0.5 * damage_f1[1]
    return {
        "loss": total_loss / max(batches, 1),
        "land_mIoU": float(land_iou.mean()),
        "land_F1": float(land_f1.mean()),
        "damage_IoU": float(damage_iou[1]),
        "damage_F1": float(damage_f1[1]),
        "val_combined_F1": float(combined_f1),
    }


def save_checkpoint(path: Path, model: LGDNet, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = make_dataset(config, "train", train=True)
    val_data = make_dataset(config, "val", train=False)
    train_loader = DataLoader(
        train_data,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["train"].get("val_batch_size", config["train"]["batch_size"]),
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = LGDNet(
        num_land_classes=config["data"]["num_land_classes"],
        decoder_channels=config["model"]["decoder_channels"],
        use_lsgm=config["model"]["use_lsgm"],
        host_class_ids=tuple(config["model"]["host_class_ids"]),
        non_host_class_ids=tuple(config["model"]["non_host_class_ids"]),
        pretrained_backbone=config["model"].get("pretrained_backbone", True),
    ).to(device)

    criterion = LGDNetLoss(
        damage_weight=config["loss"]["damage_weight"],
        land_weight=config["loss"]["land_weight"],
        damage_class_weights=config["loss"].get("damage_class_weights"),
        land_class_weights=config["loss"].get("land_class_weights"),
        num_land_classes=config["data"]["num_land_classes"],
    )
    base_optimizer = torch.optim.AdamW(parameter_groups(model, config))
    optimizer = Lookahead(
        base_optimizer,
        k=config["optimizer"].get("lookahead_k", 5),
        alpha=config["optimizer"].get("lookahead_alpha", 0.5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer,
        T_max=config["train"]["epochs"],
        eta_min=config["scheduler"].get("min_lr", 1e-6),
    )

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = config["train"].get("monitor", "val_combined_F1")
    best_score = float("-inf")

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images, land, damage = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = criterion(outputs, land, damage)
            losses["total"].backward()
            optimizer.step()
            running_loss += float(losses["total"].detach())

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            config["data"]["num_land_classes"],
            config["eval"].get("damage_threshold", 0.5),
        )
        score = val_metrics[monitor]
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_combined_F1={score:.6f} "
            f"land_F1={val_metrics['land_F1']:.6f} damage_F1={val_metrics['damage_F1']:.6f}"
        )

        save_checkpoint(output_dir / "last.ckpt", model, base_optimizer, epoch, val_metrics)
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.ckpt", model, base_optimizer, epoch, val_metrics)


if __name__ == "__main__":
    main()
