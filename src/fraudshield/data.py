"""Dataset loading and validation contracts."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{index}" for index in range(1, 29)),
    "Amount",
)
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)


class DatasetSchemaError(ValueError):
    """Raised when transaction data violates the modeling schema."""


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required transaction columns and return the input frame."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetSchemaError(f"Missing required columns: {', '.join(missing)}")

    target_values = set(frame[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise DatasetSchemaError("Class must be a binary target containing only 0 and 1")
    return frame
