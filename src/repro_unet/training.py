from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


class WeightedBinaryCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted_loss = loss * weights
        return weighted_loss.sum() / weights.sum().clamp_min(1e-8)


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    prediction_sum = predictions.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))
    union = prediction_sum + target_sum - intersection

    dice = ((2.0 * intersection + 1e-6) / (prediction_sum + target_sum + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    accuracy = (predictions == targets).float().mean().item()
    return {"dice": float(dice), "iou": float(iou), "pixel_accuracy": float(accuracy)}


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: WeightedBinaryCrossEntropy,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    running_metrics: list[dict[str, float]] = []

    for images, targets, weights in loader:
        images = images.to(device)
        targets = targets.to(device)
        weights = weights.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets, weights)

        if is_training:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        running_metrics.append(segmentation_metrics(logits.detach(), targets))

    return {
        "loss": running_loss / max(len(loader), 1),
        "dice": float(np.mean([item["dice"] for item in running_metrics])),
        "iou": float(np.mean([item["iou"] for item in running_metrics])),
        "pixel_accuracy": float(np.mean([item["pixel_accuracy"] for item in running_metrics])),
    }


def run_training(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, float]:
    criterion = WeightedBinaryCrossEntropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)
    model.to(device)

    best_state = None
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    elapsed = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), elapsed


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    criterion = WeightedBinaryCrossEntropy()
    return _run_epoch(model, loader, criterion, optimizer=None, device=device)


@torch.no_grad()
def save_prediction_figure(
    model_name: str,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    output_path: Path,
    device: torch.device,
) -> None:
    model.eval()
    images, targets, _ = next(iter(loader))
    images = images.to(device)
    logits = model(images)
    probabilities = torch.sigmoid(logits).cpu().numpy()
    image = images.cpu().numpy()[0, 0]
    target = targets.cpu().numpy()[0, 0]
    prediction = probabilities[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(target, cmap="gray")
    axes[1].set_title("Target")
    axes[2].imshow(prediction, cmap="magma")
    axes[2].set_title(f"{model_name} prediction")
    for axis in axes:
        axis.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_history_plot(histories: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=f"{name} train")
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(history["epoch"], history["val_dice"], label=name)

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted BCE")
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_metrics(
    results: dict[str, dict[str, float]],
    histories: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    records = []
    for name, metrics in results.items():
        record = {"model": name, **metrics}
        if name in histories:
            record["best_val_dice"] = float(histories[name]["val_dice"].max())
            record["best_val_loss"] = float(histories[name]["val_loss"].min())
        records.append(record)

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return summary
