"""PatchCore-style nearest-neighbor memory-bank anomaly detector."""

from __future__ import annotations

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


def farthest_first_coreset(
    features: torch.Tensor,
    *,
    ratio: float,
    max_candidates: int = 20_000,
    projection_dim: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Select a deterministic approximate farthest-first subset of patch embeddings."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty [N, D] tensor")
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = features.detach().cpu().float()
    if source.shape[0] > max_candidates:
        subset = torch.randperm(source.shape[0], generator=generator)[:max_candidates]
        source = source[subset]
    target_size = max(1, min(source.shape[0], round(source.shape[0] * ratio)))
    if target_size == source.shape[0]:
        return source

    if source.shape[1] > projection_dim:
        projection = torch.randn(
            source.shape[1], projection_dim, generator=generator, dtype=source.dtype
        ) / projection_dim**0.5
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
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in (0, 1]")
        if distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be positive")
        self.extractor = extractor if extractor is not None else ResNet18PatchExtractor()
        self.coreset_ratio = coreset_ratio
        self.distance_chunk_size = distance_chunk_size
        self.register_buffer("memory_bank", torch.empty(0, 0), persistent=True)
        self.threshold: float | None = None

    @torch.inference_mode()
    def fit_embeddings(self, images: torch.Tensor) -> None:
        """Build a memory bank from a normal-image tensor batch."""
        features = self.extractor(images)
        patches = _flatten_patches(features)
        bank = farthest_first_coreset(patches, ratio=self.coreset_ratio)
        self.memory_bank = bank.to(images.device)

    @torch.inference_mode()
    def fit_loader(
        self,
        loader: Iterable[Any],
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        """Build a memory bank from a loader whose first element is an image batch."""
        self.to(device)
        patches: list[torch.Tensor] = []
        for batch in loader:
            images = batch[0].to(device)
            patches.append(_flatten_patches(self.extractor(images)).cpu())
        if not patches:
            raise ValueError("training loader produced no images")
        all_patches = torch.cat(patches, dim=0)
        self.memory_bank = farthest_first_coreset(all_patches, ratio=self.coreset_ratio).to(device)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> Prediction:
        if self.memory_bank.numel() == 0:
            raise RuntimeError("memory bank is empty; fit the model before prediction")
        feature_map = self.extractor(images)
        batch, _, height, width = feature_map.shape
        patches = _flatten_patches(feature_map)
        nearest = self._nearest_distance(patches)
        patch_map = nearest.reshape(batch, height, width)[:, None, ...]
        anomaly_map = F.interpolate(
            patch_map,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        image_scores = anomaly_map.flatten(1).amax(dim=1)
        return Prediction(image_scores=image_scores, anomaly_maps=anomaly_map)

    def _nearest_distance(self, patches: torch.Tensor) -> torch.Tensor:
        output: list[torch.Tensor] = []
        bank = self.memory_bank.to(patches.device)
        for start in range(0, patches.shape[0], self.distance_chunk_size):
            chunk = patches[start : start + self.distance_chunk_size]
            output.append(torch.cdist(chunk, bank).amin(dim=1))
        return torch.cat(output, dim=0)

    def calibrate_threshold(self, normal_scores: torch.Tensor, *, quantile: float = 0.995) -> float:
        if normal_scores.numel() == 0:
            raise ValueError("normal_scores cannot be empty")
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        calibrated = torch.quantile(normal_scores.detach().float().cpu(), quantile)
        self.threshold = float(calibrated.item())
        return self.threshold

    def save(self, path: str | Path) -> None:
        if self.memory_bank.numel() == 0:
            raise RuntimeError("cannot save an unfitted model")
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
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu") -> "PatchMemory":
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("extractor") != "resnet18_multiscale":
            raise ValueError("unsupported checkpoint extractor")
        extractor = ResNet18PatchExtractor(pretrained=False)
        extractor.load_state_dict(payload["extractor_state"])
        model = cls(
            extractor,
            coreset_ratio=float(payload["coreset_ratio"]),
            distance_chunk_size=int(payload["distance_chunk_size"]),
        )
        model.memory_bank = payload["memory_bank"].to(device)
        threshold = payload.get("threshold")
        model.threshold = None if threshold is None else float(threshold)
        model.to(device)
        return model


def _flatten_patches(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError("extractor must return [B, C, H, W] features")
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
