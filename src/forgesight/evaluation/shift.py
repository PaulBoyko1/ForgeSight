"""Distribution-shift grouped reporting."""

from __future__ import annotations

import pandas as pd

from forgesight.evaluation.metrics import binary_anomaly_metrics


def metrics_by_domain(frame: pd.DataFrame, *, reference_domain: str | None = None) -> pd.DataFrame:
    """Compute image metrics by acquisition domain and optional AUROC degradation."""
    required = {"domain", "label", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing evaluation columns: {sorted(missing)}")
    rows: list[dict[str, float | str | int]] = []
    for domain, group in frame.groupby("domain", sort=True):
        if group["label"].nunique() < 2:
            continue
        metrics = binary_anomaly_metrics(group["label"].to_numpy(), group["score"].to_numpy())
        rows.append({"domain": str(domain), "samples": len(group), **metrics})
    result = pd.DataFrame(rows)
    if reference_domain is not None and not result.empty:
        match = result[result["domain"] == reference_domain]
        if match.empty:
            raise ValueError(f"reference domain {reference_domain!r} has no evaluable rows")
        reference = float(match.iloc[0]["auroc"])
        result["auroc_degradation"] = reference - result["auroc"]
    return result
