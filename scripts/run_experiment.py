from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repro_unet import (  # noqa: E402
    BinarySegmentationDataset,
    ExperimentConfig,
    UNetOriginal,
    build_synthetic_dataset,
    count_parameters,
    evaluate_model,
    run_training,
    set_seed,
    summarize_metrics,
)
from repro_unet.training import save_history_plot, save_prediction_figure  # noqa: E402


def load_reference_model(reference_path: Path) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location("reference_unet_module", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reference implementation from {reference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Unet(channels=[1, 64, 128, 256, 512, 1024], no_classes=1)


def infer_output_hw(model: torch.nn.Module, image_size: int) -> tuple[int, int]:
    with torch.no_grad():
        dummy = torch.zeros(1, 1, image_size, image_size)
        output = model(dummy)
    return int(output.shape[-2]), int(output.shape[-1])


def save_dataset_overview(images: np.ndarray, masks: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.5), constrained_layout=True)
    for axis, idx in zip(axes.flat, [0, 1, 2, 3, 4, 5]):
        axis.imshow(images[idx], cmap="gray")
        axis.imshow(masks[idx], cmap="Reds", alpha=0.22)
        axis.set_title(f"Sample {idx}")
        axis.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_dataset(dataset_bundle: dict[str, np.ndarray], config: ExperimentConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_dataset(config)

    for split_name, split_slice in splits.items():
        split_dir = output_dir / split_name
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        weight_dir = split_dir / "weights"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        weight_dir.mkdir(parents=True, exist_ok=True)

        images = dataset_bundle["images"][split_slice]
        masks = dataset_bundle["masks"][split_slice]
        weights = dataset_bundle["weights"][split_slice]

        for index, (image, mask, weight) in enumerate(zip(images, masks, weights)):
            stem = f"{split_name}_{index:03d}"
            Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(image_dir / f"{stem}.png")
            Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8)).save(mask_dir / f"{stem}.png")

            normalized_weight = weight / max(float(weight.max()), 1e-8)
            Image.fromarray(np.clip(normalized_weight * 255.0, 0, 255).astype(np.uint8)).save(
                weight_dir / f"{stem}.png"
            )

    manifest = {
        "image_size": config.image_size,
        "total_samples": config.total_samples,
        "train_samples": config.train_samples,
        "val_samples": config.val_samples,
        "test_samples": config.test_samples,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def split_dataset(config: ExperimentConfig) -> dict[str, slice]:
    train_end = config.train_samples
    val_end = train_end + config.val_samples
    test_end = val_end + config.test_samples
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, test_end),
    }


def build_loaders(
    dataset_bundle: dict[str, np.ndarray],
    target_hw: tuple[int, int],
    config: ExperimentConfig,
    batch_size: int,
) -> dict[str, DataLoader]:
    splits = split_dataset(config)
    loaders: dict[str, DataLoader] = {}
    for split_name, split_slice in splits.items():
        dataset = BinarySegmentationDataset(
            images=dataset_bundle["images"][split_slice],
            masks=dataset_bundle["masks"][split_slice],
            weights=dataset_bundle["weights"][split_slice],
            target_hw=target_hw,
            augment=split_name == "train",
            seed=config.seed + len(split_name),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=0,
        )
    return loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a scratch U-Net to the uploaded reference implementation.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(7)
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    config = ExperimentConfig(image_size=args.image_size)
    dataset_bundle = build_synthetic_dataset(config)
    save_dataset_overview(dataset_bundle["images"], dataset_bundle["masks"], artifacts_dir / "dataset_overview.png")
    export_dataset(dataset_bundle, config=config, output_dir=artifacts_dir / "dataset")

    scratch_model = UNetOriginal()
    reference_model = load_reference_model(
        ROOT / "official_repo" / "UNet_Biomedical_Image_Segmentation-main" / "UNet.py"
    )
    output_hw = infer_output_hw(scratch_model, config.image_size)
    reference_output_hw = infer_output_hw(reference_model, config.image_size)
    if output_hw != reference_output_hw:
        raise RuntimeError(f"Output mismatch: scratch={output_hw}, reference={reference_output_hw}")

    loaders = build_loaders(dataset_bundle, output_hw, config=config, batch_size=args.batch_size)
    device = torch.device(args.device)

    results: dict[str, dict[str, float]] = {}
    histories = {}
    model_objects = {
        "scratch_unet": scratch_model,
        "reference_repo_unet": reference_model,
    }

    comparison_rows = []
    for model_name, model in model_objects.items():
        trained_model, history, elapsed_seconds = run_training(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
        )
        metrics = evaluate_model(trained_model, loaders["test"], device=device)
        metrics["parameter_count"] = count_parameters(trained_model)
        metrics["training_seconds"] = elapsed_seconds
        results[model_name] = metrics
        histories[model_name] = history
        history.to_csv(artifacts_dir / f"{model_name}_history.csv", index=False)
        save_prediction_figure(
            model_name=model_name,
            model=trained_model,
            loader=loaders["test"],
            output_path=artifacts_dir / f"{model_name}_prediction.png",
            device=device,
        )
        comparison_rows.append(
            {
                "model": model_name,
                "test_loss": metrics["loss"],
                "test_dice": metrics["dice"],
                "test_iou": metrics["iou"],
                "test_pixel_accuracy": metrics["pixel_accuracy"],
                "parameter_count": metrics["parameter_count"],
                "training_seconds": metrics["training_seconds"],
            }
        )

    summary = summarize_metrics(results, histories, artifacts_dir)
    save_history_plot(histories, artifacts_dir / "training_curves.png")

    experiment_metadata = {
        "paper_title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "image_size": args.image_size,
            "output_hw": output_hw,
            "total_samples": config.total_samples,
            "train_samples": config.train_samples,
            "val_samples": config.val_samples,
            "test_samples": config.test_samples,
            "device": str(device),
        },
        "results": comparison_rows,
    }
    with (artifacts_dir / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_metadata, handle, indent=2)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
