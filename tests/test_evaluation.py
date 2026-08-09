import numpy as np
import pandas as pd
import pytest

from forgesight.evaluation.metrics import binary_anomaly_metrics, pixel_auroc
from forgesight.evaluation.shift import metrics_by_domain


def test_binary_metrics_perfect_ranking() -> None:
    metrics = binary_anomaly_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_pixel_auroc() -> None:
    result = pixel_auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        (np.array([0, 2]), np.array([0.1, 0.9])),
        (np.array([0, 1]), np.array([0.1, np.nan])),
        (np.array([0.5, 1.0]), np.array([0.1, 0.9])),
    ],
)
def test_metrics_reject_malformed_inputs(labels: np.ndarray, scores: np.ndarray) -> None:
    with pytest.raises(ValueError):
        binary_anomaly_metrics(labels, scores)
    with pytest.raises(ValueError):
        pixel_auroc(labels, scores)


def test_domain_report_tracks_degradation() -> None:
    frame = pd.DataFrame(
        {
            "domain": ["base"] * 4 + ["shift"] * 4,
            "label": [0, 0, 1, 1] * 2,
            "score": [0.1, 0.2, 0.8, 0.9, 0.4, 0.1, 0.5, 0.6],
        }
    )
    result = metrics_by_domain(frame, reference_domain="base")
    base = result[result["domain"] == "base"].iloc[0]
    shift = result[result["domain"] == "shift"].iloc[0]
    assert base["auroc_degradation"] == pytest.approx(0.0)
    assert shift["auroc_degradation"] >= 0.0
