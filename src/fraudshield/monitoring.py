"""Feature and prediction drift monitoring using population stability index (PSI)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite, log
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DriftStatus = Literal["ok", "warning", "critical"]

DEFAULT_PSI_WARNING_THRESHOLD = 0.10
DEFAULT_PSI_CRITICAL_THRESHOLD = 0.25
DEFAULT_MONITORING_BINS = 10
PSI_EPSILON = 1e-6


@dataclass(frozen=True)
class AlertThresholds:
    """Configurable PSI alert thresholds."""

    warning: float = DEFAULT_PSI_WARNING_THRESHOLD
    critical: float = DEFAULT_PSI_CRITICAL_THRESHOLD

    def __post_init__(self) -> None:
        if not isfinite(self.warning) or not isfinite(self.critical):
            raise ValueError("alert thresholds must be finite")
        if self.warning < 0.0 or self.critical < 0.0:
            raise ValueError("alert thresholds must be non-negative")
        if self.warning >= self.critical:
            raise ValueError("warning threshold must be lower than critical threshold")


@dataclass(frozen=True)
class StabilityBucket:
    """One stable reference-defined PSI bucket."""

    lower_bound: float
    upper_bound: float
    reference_count: int
    current_count: int
    reference_proportion: float
    current_proportion: float
    psi_component: float

    def to_report(self) -> dict[str, object]:
        """Return JSON-compatible bucket evidence."""
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "reference_proportion": self.reference_proportion,
            "current_proportion": self.current_proportion,
            "psi_component": self.psi_component,
        }


@dataclass(frozen=True)
class PopulationStabilityResult:
    """PSI result for one monitored numeric distribution."""

    feature: str
    bins: int
    reference_count: int
    current_count: int
    psi: float
    buckets: tuple[StabilityBucket, ...]

    def status(self, thresholds: AlertThresholds) -> DriftStatus:
        """Classify PSI against alert thresholds."""
        if self.psi >= thresholds.critical:
            return "critical"
        if self.psi >= thresholds.warning:
            return "warning"
        return "ok"

    def to_report(self, *, kind: str, thresholds: AlertThresholds) -> dict[str, object]:
        """Return JSON-compatible metric evidence."""
        status = self.status(thresholds)
        name_prefix = "prediction" if kind == "prediction_score" else kind
        return {
            "name": f"{name_prefix}:{self.feature}",
            "kind": kind,
            "feature": self.feature,
            "statistic": "population_stability_index",
            "psi": self.psi,
            "status": status,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "bins": self.bins,
            "buckets": [bucket.to_report() for bucket in self.buckets],
        }


def compute_population_stability(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = DEFAULT_MONITORING_BINS,
    feature_name: str | None = None,
    epsilon: float = PSI_EPSILON,
) -> PopulationStabilityResult:
    """Compute PSI using bucket edges learned only from the reference window."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if epsilon <= 0.0 or not isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive value")
    name = feature_name or str(reference.name or "value")
    reference_values = _finite_numeric_values(reference, window_name="reference")
    current_values = _finite_numeric_values(current, window_name="current")
    edges = _reference_edges(reference_values, bins)
    reference_counts = _bucket_counts(reference_values, edges)
    current_counts = _bucket_counts(current_values, edges)
    reference_total = int(reference_counts.sum())
    current_total = int(current_counts.sum())
    buckets: list[StabilityBucket] = []
    psi = 0.0
    for index in range(bins):
        reference_count = int(reference_counts[index])
        current_count = int(current_counts[index])
        reference_proportion = _smoothed_proportion(reference_count, reference_total, bins, epsilon)
        current_proportion = _smoothed_proportion(current_count, current_total, bins, epsilon)
        component = (current_proportion - reference_proportion) * log(
            current_proportion / reference_proportion
        )
        psi += component
        buckets.append(
            StabilityBucket(
                lower_bound=float(edges[index]),
                upper_bound=float(edges[index + 1]),
                reference_count=reference_count,
                current_count=current_count,
                reference_proportion=reference_proportion,
                current_proportion=current_proportion,
                psi_component=float(component),
            )
        )
    return PopulationStabilityResult(
        feature=name,
        bins=bins,
        reference_count=len(reference_values),
        current_count=len(current_values),
        psi=float(psi),
        buckets=tuple(buckets),
    )


def generate_monitoring_report(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: tuple[str, ...],
    score_column: str,
    thresholds: AlertThresholds | None = None,
    bins: int = DEFAULT_MONITORING_BINS,
) -> dict[str, object]:
    """Generate a versioned monitoring report for feature and prediction-score drift."""
    if not feature_columns:
        raise ValueError("at least one feature column is required")
    threshold_config = thresholds or AlertThresholds()
    required_columns = (*feature_columns, score_column)
    missing = sorted(
        {
            column
            for column in required_columns
            if column not in reference.columns or column not in current.columns
        }
    )
    if missing:
        raise ValueError(f"missing required monitoring columns: {', '.join(missing)}")

    metrics: list[dict[str, object]] = []
    for column in feature_columns:
        result = compute_population_stability(
            reference[column], current[column], bins=bins, feature_name=column
        )
        metrics.append(result.to_report(kind="feature", thresholds=threshold_config))
    score_result = compute_population_stability(
        reference[score_column], current[score_column], bins=bins, feature_name=score_column
    )
    metrics.append(score_result.to_report(kind="prediction_score", thresholds=threshold_config))

    alerts: list[dict[str, object]] = []
    for metric in metrics:
        if metric["status"] != "ok":
            psi_value = metric["psi"]
            if not isinstance(psi_value, int | float):
                raise TypeError("monitoring metric psi must be numeric")
            alerts.append(
                {
                    "name": str(metric["name"]),
                    "kind": str(metric["kind"]),
                    "psi": float(psi_value),
                    "status": str(metric["status"]),
                }
            )
    overall_status = _overall_status(tuple(str(metric["status"]) for metric in metrics))

    return {
        "schema_version": 1,
        "statistic": "population_stability_index",
        "status": overall_status,
        "thresholds": {"warning": threshold_config.warning, "critical": threshold_config.critical},
        "windows": {"reference_rows": len(reference), "current_rows": len(current)},
        "metrics": metrics,
        "alerts": alerts,
        "limitations": [
            "PSI compares distribution shift only; it does not prove model-performance "
            "degradation without current labels.",
            "Reference-defined numeric bins should be refreshed only through a governed "
            "model-monitoring process.",
        ],
    }


def write_monitoring_report(report: dict[str, object], report_path: Path) -> None:
    """Write monitoring evidence as deterministic JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _finite_numeric_values(series: pd.Series, *, window_name: str) -> NDArray[np.float64]:
    if len(series) == 0:
        raise ValueError(f"{window_name} window must not be empty")
    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{window_name} window must contain finite numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{window_name} window must contain finite numeric values")
    return values


def _reference_edges(reference_values: NDArray[np.float64], bins: int) -> NDArray[np.float64]:
    lower = float(np.min(reference_values))
    upper = float(np.max(reference_values))
    if lower == upper:
        padding = max(abs(lower) * 0.001, 0.001)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, bins + 1, dtype=float)


def _smoothed_proportion(count: int, total: int, bins: int, epsilon: float) -> float:
    return (count + epsilon) / (total + epsilon * bins)


def _bucket_counts(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int_]:
    bucket_indexes = np.searchsorted(edges[1:-1], values, side="right")
    bucket_indexes = np.clip(bucket_indexes, 0, len(edges) - 2)
    return np.bincount(bucket_indexes, minlength=len(edges) - 1)


def _overall_status(statuses: tuple[str, ...]) -> DriftStatus:
    if "critical" in statuses:
        return "critical"
    if "warning" in statuses:
        return "warning"
    return "ok"
