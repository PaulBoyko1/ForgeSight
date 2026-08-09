"""PatchCore-style nearest-neighbor memory-bank anomaly detector."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from forgesight.models.features import ResNet18PatchExtractor


@dataclass(frozen=True)
class Prediction:
    image_scores: torch.Tensor
    anomaly_maps: torch.Tensor


def _finite_ratio(value: float, name: str) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return ratio


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_feature_matrix(features: torch.Tensor, *, name: str) -> None:
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] tensor")
    if not features.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_images(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("images must be a non-empty [B, C, H, W] tensor")
    if not images.is_floating_point():
        raise ValueError("images must use a floating-point dtype")
    if not torch.isfinite(images).all():
        raise ValueError("images must contain only finite values")
    return images


def farthest_first_coreset(
    features: torch.Tensor,
    *,
    ratio: float,
    max_candidates: int = 20_000,
    projection_dim: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Select a deterministic approximate farthest-first subset of patch embeddings."""
    _validate_feature_matrix(features, name="features")
    ratio_value = _finite_ratio(ratio, "ratio")
    max_candidates_value = _positive_int(max_candidates, "max_candidates")
    projection_dim_value = _positive_int(projection_dim, "projection_dim")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = features.detach().cpu().float()
    if source.shape[0] > max_candidates_value:
        subset = torch.randperm(source.shape[0], generator=generator)[:max_candidates_value]
        source = source[subset]
    target_size = max(1, min(source.shape[0], round(source.shape[0] * ratio_value)))
    if target_size == source.shape[0]:
        return source

    if source.shape[1] > projection_dim_value:
        projection = torch.randn(
            source.shape[1],
            projection_dim_value,
            generator=generator,
            dtype=source.dtype,
        ) / projection_dim_value**0.5
        search_space = source @ projection
    else:
        search_space = source

    center = search_space.mean(dim=0, keepdim=True)
    first = int(torch.argmax(torch.sum((search_space - center) ** 2, dim=1)).item())
    selected = [first]
    min_distance = torch.sum((search_space - search_space[first]) ** 2, dim=1)
    for _ in range(1, target_size):
        index = int(torch.argmax(min_distance).item())
        selected.append(index)
        distance = torch.sum((search_space - search_space[index]) ** 2, dim=1)
        min_distance = torch.minimum(min_distance, distance)
    return source[torch.tensor(selected, dtype=torch.long)]


class PatchMemory(nn.Module):
    """Normal-patch memory model with spatial anomaly-map output."""

    def __init__(
        self,
        extractor: nn.Module | None = None,
        *,
        coreset_ratio: float = 0.05,
        distance_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.extractor = extractor if extractor is not None else ResNet18PatchExtractor()
        self.coreset_ratio = _finite_ratio(coreset_ratio, "coreset_ratio")
        self.distance_chunk_size = _positive_int(distance_chunk_size, "distance_chunk_size")
        self.register_buffer("memory_bank", torch.empty(0, 0), persistent=True)
        self.threshold: float | None = None
        self.eval()

    @torch.inference_mode()
    def fit_embeddings(self, images: torch.Tensor) -> None:
        """Build a memory bank from a normal-image tensor batch."""
        image_batch = _validate_images(images)
        self.eval()
        features = self.extractor(image_batch)
        patches = _flatten_patches(features)
        bank = farthest_first_coreset(patches, ratio=self.coreset_ratio)
        self.memory_bank = bank.to(features.device)

    @torch.inference_mode()
    def fit_loader(
        self,
        loader: Iterable[Any],
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        """Build a memory bank from a loader whose first element is an image batch."""
        self.to(device)
        self.eval()
        patches: list[torch.Tensor] = []
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or not batch:
                raise ValueError("training loader must yield batches beginning with images")
            images = _validate_images(batch[0]).to(device)
            patches.append(_flatten_patches(self.extractor(images)).cpu())
        if not patches:
            raise ValueError("training loader produced no images")
        all_patches = torch.cat(patches, dim=0)
        self.memory_bank = farthest_first_coreset(all_patches, ratio=self.coreset_ratio).to(device)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> Prediction:
        image_batch = _validate_images(images)
        self._validate_memory_bank()
        self.eval()
        feature_map = self.extractor(image_batch)
        batch, _, height, width = feature_map.shape
        patches = _flatten_patches(feature_map)
        nearest = self._nearest_distance(patches)
        patch_map = nearest.reshape(batch, height, width)[:, None, ...]
        anomaly_map = F.interpolate(
            patch_map,
            size=image_batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        image_scores = anomaly_map.flatten(1).amax(dim=1)
        return Prediction(image_scores=image_scores, anomaly_maps=anomaly_map)

    def _validate_memory_bank(self) -> None:
        _validate_feature_matrix(self.memory_bank, name="memory bank")

    def _nearest_distance(self, patches: torch.Tensor) -> torch.Tensor:
        _validate_feature_matrix(patches, name="patches")
        self._validate_memory_bank()
        bank = self.memory_bank
        if patches.shape[1] != bank.shape[1]:
            raise ValueError("patch embedding size does not match the memory bank")
        if bank.device != patches.device:
            bank = bank.to(patches.device)
        output: list[torch.Tensor] = []
        for start in range(0, patches.shape[0], self.distance_chunk_size):
            chunk = patches[start : start + self.distance_chunk_size]
            output.append(torch.cdist(chunk, bank).amin(dim=1))
        return torch.cat(output, dim=0)

    def calibrate_threshold(self, normal_scores: torch.Tensor, *, quantile: float = 0.995) -> float:
        if not isinstance(normal_scores, torch.Tensor):
            raise TypeError("normal_scores must be a torch.Tensor")
        scores = normal_scores.detach().float().cpu().flatten()
        if scores.numel() == 0 or not torch.isfinite(scores).all():
            raise ValueError("normal_scores must be non-empty and finite")
        quantile_value = _finite_ratio(quantile, "quantile")
        if quantile_value >= 1.0:
            raise ValueError("quantile must be in (0, 1)")
        calibrated = torch.quantile(scores, quantile_value)
        self.threshold = float(calibrated.item())
        return self.threshold

    def save(self, path: str | Path) -> None:
        self._validate_memory_bank()
        if not isinstance(self.extractor, ResNet18PatchExtractor):
            raise RuntimeError("portable checkpoints currently require ResNet18PatchExtractor")
        payload = {
            "version": 1,
            "extractor": "resnet18_multiscale",
            "extractor_state": self.extractor.state_dict(),
            "memory_bank": self.memory_bank.detach().cpu(),
            "threshold": self.threshold,
            "coreset_ratio": self.coreset_ratio,
            "distance_chunk_size": self.distance_chunk_size,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu") -> PatchMemory:
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported checkpoint format")
        if payload.get("extractor") != "resnet18_multiscale":
            raise ValueError("unsupported checkpoint extractor")

        try:
            coreset_ratio = _finite_ratio(payload["coreset_ratio"], "checkpoint coreset_ratio")
            distance_chunk_size = _positive_int(
                payload["distance_chunk_size"],
                "checkpoint distance_chunk_size",
            )
            memory_bank = payload["memory_bank"]
            extractor_state = payload["extractor_state"]
        except KeyError as exc:
            raise ValueError(f"checkpoint is missing {exc.args[0]!r}") from exc
        _validate_feature_matrix(memory_bank, name="checkpoint memory bank")
        if not isinstance(extractor_state, dict):
            raise ValueError("checkpoint extractor_state must be a state dictionary")

        threshold_value = payload.get("threshold")
        threshold = None if threshold_value is None else _finite_threshold(threshold_value)

        extractor = ResNet18PatchExtractor(pretrained=False)
        extractor.load_state_dict(extractor_state)
        model = cls(
            extractor,
            coreset_ratio=coreset_ratio,
            distance_chunk_size=distance_chunk_size,
        )
        model.memory_bank = memory_bank.detach().to(device)
        model.threshold = threshold
        model.to(device)
        model.eval()
        return model


def _finite_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("checkpoint threshold must be a finite number")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("checkpoint threshold must be finite")
    return threshold


def _flatten_patches(features: torch.Tensor) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise TypeError("extractor output must be a torch.Tensor")
    if features.ndim != 4 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("extractor must return a non-empty [B, C, H, W] tensor")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("extractor output must be finite floating-point features")
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
