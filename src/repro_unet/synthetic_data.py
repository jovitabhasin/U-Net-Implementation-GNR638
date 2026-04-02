from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
)
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: int = 256
    total_samples: int = 30
    train_samples: int = 20
    val_samples: int = 5
    test_samples: int = 5
    w0: float = 10.0
    sigma: float = 5.0
    seed: int = 7


def min_max_scale(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_val) / (max_val - min_val)).astype(np.float32)


def random_centers(rng: np.random.Generator, image_size: int, count: int) -> np.ndarray:
    margin = max(12, image_size // 16)
    min_distance = max(10, image_size // 14)
    centers: list[tuple[int, int]] = []
    attempts = 0

    while len(centers) < count and attempts < count * 200:
        attempts += 1
        row = int(rng.integers(margin, image_size - margin))
        col = int(rng.integers(margin, image_size - margin))
        if all((row - r) ** 2 + (col - c) ** 2 >= min_distance**2 for r, c in centers):
            centers.append((row, col))

    if len(centers) < count:
        while len(centers) < count:
            centers.append(
                (
                    int(rng.integers(margin, image_size - margin)),
                    int(rng.integers(margin, image_size - margin)),
                )
            )

    return np.asarray(centers, dtype=np.float32)


def elastic_deform(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = image.shape
    dx = gaussian_filter((rng.random(shape) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha
    dy = gaussian_filter((rng.random(shape) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha

    coords_y, coords_x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    warped_indices = np.vstack([(coords_y + dy).ravel(), (coords_x + dx).ravel()])

    warped_image = map_coordinates(image, warped_indices, order=1, mode="reflect").reshape(shape)
    warped_mask = map_coordinates(mask.astype(np.float32), warped_indices, order=0, mode="reflect").reshape(shape)
    return warped_image.astype(np.float32), warped_mask.astype(np.float32)


def render_sample(rng: np.random.Generator, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    center_count = int(rng.integers(10, 18))
    centers = random_centers(rng, image_size=image_size, count=center_count)
    rows, cols = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")

    centers_r = centers[:, 0][:, None, None]
    centers_c = centers[:, 1][:, None, None]
    scale_r = rng.uniform(0.8, 1.25, size=(center_count, 1, 1))
    scale_c = rng.uniform(0.8, 1.25, size=(center_count, 1, 1))
    distances = np.sqrt(((rows - centers_r) / scale_r) ** 2 + ((cols - centers_c) / scale_c) ** 2)

    regions = np.argmin(distances, axis=0).astype(np.int32) + 1
    boundaries = np.zeros_like(regions, dtype=bool)
    boundaries[1:, :] |= regions[1:, :] != regions[:-1, :]
    boundaries[:, 1:] |= regions[:, 1:] != regions[:, :-1]
    boundaries = binary_dilation(boundaries, iterations=int(rng.integers(2, 5)))

    mask = (~boundaries).astype(np.float32)
    cell_intensity = rng.uniform(0.45, 0.9, size=center_count + 1).astype(np.float32)
    cell_intensity[0] = rng.uniform(0.02, 0.12)

    image = cell_intensity[regions]
    image[boundaries] = rng.uniform(0.0, 0.08)

    illumination = gaussian_filter(rng.normal(size=image.shape).astype(np.float32), sigma=image_size / 10.0)
    texture = gaussian_filter(rng.normal(size=image.shape).astype(np.float32), sigma=1.2)
    image = image + 0.15 * min_max_scale(illumination) + 0.08 * min_max_scale(texture)
    image = gaussian_filter(image, sigma=rng.uniform(0.6, 1.2))
    image = min_max_scale(image)

    alpha = image_size * rng.uniform(0.03, 0.07)
    sigma = image_size * rng.uniform(0.03, 0.06)
    image, mask = elastic_deform(image, mask, rng=rng, alpha=alpha, sigma=sigma)
    image = min_max_scale(image + rng.normal(0.0, 0.035, size=image.shape).astype(np.float32))
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


def compute_weight_map(mask: np.ndarray, w0: float, sigma: float) -> np.ndarray:
    mask = (mask > 0.5).astype(np.uint8)
    height, width = mask.shape
    weights = np.zeros((height, width), dtype=np.float32)

    unique, counts = np.unique(mask, return_counts=True)
    freq = counts.astype(np.float32) / mask.size
    max_freq = float(freq.max())
    class_weight_map = np.zeros_like(weights)
    for cls, cls_freq in zip(unique, freq):
        class_weight_map[mask == cls] = max_freq / max(cls_freq, 1e-8)

    connected, num_components = label(mask)
    if num_components > 1:
        distance_stack = np.zeros((num_components, height, width), dtype=np.float32)
        for idx in range(1, num_components + 1):
            distance_stack[idx - 1] = distance_transform_edt(connected != idx).astype(np.float32)
        sorted_distances = np.sort(distance_stack, axis=0)
        d1 = sorted_distances[0]
        d2 = sorted_distances[1]
        border_term = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * sigma**2))
        weights = border_term * (mask == 0).astype(np.float32)

    return (weights + class_weight_map).astype(np.float32)


def center_crop(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = array.shape[-2:]
    target_h, target_w = target_hw
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return array[..., top : top + target_h, left : left + target_w]


def build_synthetic_dataset(config: ExperimentConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    images = []
    masks = []
    weights = []

    for _ in range(config.total_samples):
        image, mask = render_sample(rng, config.image_size)
        weight_map = compute_weight_map(mask, w0=config.w0, sigma=config.sigma)
        images.append(image.astype(np.float32))
        masks.append(mask.astype(np.float32))
        weights.append(weight_map.astype(np.float32))

    return {
        "images": np.stack(images, axis=0),
        "masks": np.stack(masks, axis=0),
        "weights": np.stack(weights, axis=0),
    }


class BinarySegmentationDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        target_hw: tuple[int, int],
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.images = images.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.target_hw = target_hw
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def _maybe_augment(
        self, image: np.ndarray, mask: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.augment:
            return image, mask, weights

        if self.rng.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
            weights = np.flip(weights, axis=0)
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            weights = np.flip(weights, axis=1)
        if self.rng.random() < 0.5:
            rotations = int(self.rng.integers(0, 4))
            image = np.rot90(image, rotations)
            mask = np.rot90(mask, rotations)
            weights = np.rot90(weights, rotations)

        image = image * float(self.rng.uniform(0.9, 1.1))
        image = image + self.rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return image.copy(), mask.copy(), weights.copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.images[index]
        mask = self.masks[index]
        weights = self.weights[index]
        image, mask, weights = self._maybe_augment(image, mask, weights)
        mask = center_crop(mask, self.target_hw)
        weights = center_crop(weights, self.target_hw)

        image_tensor = torch.from_numpy(image[None, ...]).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()
        weight_tensor = torch.from_numpy(weights[None, ...]).float()
        return image_tensor, mask_tensor, weight_tensor
