"""Segmentation metrics."""

from __future__ import annotations

import torch


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()
    if pred.numel() != target.numel():
        raise ValueError("pred and target must contain the same number of pixels")
    # Exclude ignore labels first, then enforce the canonical class range.
    valid = target != ignore_index
    valid &= (target >= 0) & (target < num_classes)
    if torch.any((pred[valid] < 0) | (pred[valid] >= num_classes)):
        raise ValueError(f"Prediction contains labels outside [0, {num_classes - 1}]")
    indices = num_classes * target[valid] + pred[valid]
    return torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_from_confusion(cm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    return tp / (tp + fp + fn + eps)


def f1_from_confusion(cm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    return (2 * tp) / (2 * tp + fp + fn + eps)
