"""Minimal LGDNet training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from lgdnet.data import LCAEQDataset
from lgdnet.losses import LGDNetLoss
from lgdnet.models import LGDNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LGDNet on LCA-EQ-style manifests.")
    parser.add_argument("--config", default="configs/lgdnet_lca_eq.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = LCAEQDataset(config["data"]["train_manifest"], root=".")
    train_loader = DataLoader(
        train_data,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = LGDNet(
        num_land_classes=config["data"]["num_land_classes"],
        decoder_channels=config["model"]["decoder_channels"],
        use_lsgm=config["model"]["use_lsgm"],
        host_class_ids=tuple(config["model"]["host_class_ids"]),
        non_host_class_ids=tuple(config["model"]["non_host_class_ids"]),
    ).to(device)

    criterion = LGDNetLoss(
        damage_weight=config["loss"]["damage_weight"],
        land_weight=config["loss"]["land_weight"],
        aux_weight=config["loss"]["aux_weight"],
        damage_class_weights=config["loss"]["damage_class_weights"],
        num_land_classes=config["data"]["num_land_classes"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["head_lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["train"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            land = batch["land_mask"].to(device)
            damage = batch["damage_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = criterion(outputs, land, damage)
            losses["total"].backward()
            optimizer.step()
            running_loss += float(losses["total"].detach())

        mean_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")

    torch.save(model.state_dict(), output_dir / "lgdnet_final.pt")


if __name__ == "__main__":
    main()
