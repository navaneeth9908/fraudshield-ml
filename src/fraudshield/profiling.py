"""Deterministic, versioned data-quality profiling evidence."""

from __future__ import annotations

from pathlib import Path
from typing import SupportsFloat

import pandas as pd

from fraudshield.data import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    DatasetIntegrityError,
    validate_transactions,
)

DATA_QUALITY_SCHEMA_VERSION = 1
DATA_QUALITY_REPORT_TYPE = "fraudshield_data_quality"


def generate_data_quality_report(
    frame: pd.DataFrame,
    *,
    dataset_path: Path,
    csv_sha256: str,
    expected_csv_sha256: str,
    dataset_slug: str,
) -> dict[str, object]:
    """Validate a transaction frame and return deterministic profiling evidence."""
    if csv_sha256.lower() != expected_csv_sha256.lower():
        raise DatasetIntegrityError(
            f"SHA-256 mismatch for {dataset_path.name}: expected {expected_csv_sha256.lower()}, "
            f"got {csv_sha256.lower()}"
        )
    validate_transactions(frame)

    row_count = len(frame)
    fraud_count = int(frame[TARGET_COLUMN].sum())
    genuine_count = row_count - fraud_count
    unexpected_columns = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))

    feature_summaries: dict[str, dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        values = frame[column]
        feature_summaries[column] = {
            "min": _stable_float(values.min()),
            "mean": _stable_float(values.mean()),
            "median": _stable_float(values.median()),
            "p95": _stable_float(values.quantile(0.95)),
            "max": _stable_float(values.max()),
        }

    fraud_fraction = fraud_count / row_count
    imbalance_ratio = genuine_count / fraud_count if fraud_count else None
    return {
        "schema_version": DATA_QUALITY_SCHEMA_VERSION,
        "report_type": DATA_QUALITY_REPORT_TYPE,
        "dataset": {
            "source": dataset_slug,
            "file_name": dataset_path.name,
            "raw_data_tracked": False,
        },
        "integrity": {
            "status": "pass",
            "algorithm": "sha256",
            "expected": expected_csv_sha256.lower(),
            "actual": csv_sha256.lower(),
        },
        "quality": {
            "status": "pass",
            "schema": {
                "status": "pass",
                "required_columns": len(REQUIRED_COLUMNS),
                "observed_columns": len(frame.columns),
                "unexpected_columns": unexpected_columns,
            },
            "missing_values": {"status": "pass", "total": 0, "columns": {}},
            "non_finite_values": {"status": "pass", "total": 0, "columns": {}},
        },
        "rows": {"total": row_count, "genuine": genuine_count, "fraud": fraud_count},
        "class_balance": {
            "genuine_fraction": genuine_count / row_count,
            "fraud_fraction": fraud_fraction,
            "imbalance_ratio_genuine_to_fraud": imbalance_ratio,
        },
        "feature_summaries": feature_summaries,
    }


def _stable_float(value: SupportsFloat) -> float:
    """Normalize floating-point noise while preserving useful profiling precision."""
    return round(float(value), 12)
