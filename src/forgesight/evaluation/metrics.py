"""Image/pixel anomaly metrics and threshold calibration helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_anomaly_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have identical shapes")
    if len(np.unique(labels)) < 2:
        raise ValueError("AUROC/AP require both normal and anomalous labels")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    masks = np.asarray(masks).astype(int).ravel()
    anomaly_maps = np.asarray(anomaly_maps).astype(float).ravel()
    if masks.shape != anomaly_maps.shape:
        raise ValueError("masks and anomaly maps must have identical flattened size")
    if len(np.unique(masks)) < 2:
        raise ValueError("pixel AUROC requires both normal and anomalous pixels")
    return float(roc_auc_score(masks, anomaly_maps))
