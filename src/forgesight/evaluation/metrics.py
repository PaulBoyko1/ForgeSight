"""Image/pixel anomaly metrics and threshold calibration helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary_labels(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        labels = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric binary values") from exc
    if labels.size == 0 or not np.all(np.isfinite(labels)):
        raise ValueError(f"{name} must be non-empty and finite")
    if not np.all(labels == np.floor(labels)) or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return labels.astype(np.int8)


def _finite_scores(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        scores = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} must be non-empty and finite")
    return scores


def binary_anomaly_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    binary_labels = _binary_labels(labels, name="labels")
    finite_scores = _finite_scores(scores, name="scores")
    if binary_labels.shape != finite_scores.shape:
        raise ValueError("labels and scores must have identical shapes")
    if len(np.unique(binary_labels)) < 2:
        raise ValueError("AUROC/AP require both normal and anomalous labels")
    return {
        "auroc": float(roc_auc_score(binary_labels, finite_scores)),
        "average_precision": float(average_precision_score(binary_labels, finite_scores)),
    }


def pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    binary_masks = _binary_labels(masks, name="masks")
    finite_maps = _finite_scores(anomaly_maps, name="anomaly_maps")
    if binary_masks.shape != finite_maps.shape:
        raise ValueError("masks and anomaly maps must have identical flattened size")
    if len(np.unique(binary_masks)) < 2:
        raise ValueError("pixel AUROC requires both normal and anomalous pixels")
    return float(roc_auc_score(binary_masks, finite_maps))
