"""Baseline model training with leakage-safe chronological evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudshield.data import FEATURE_COLUMNS, TARGET_COLUMN, validate_transactions

MetricMap = dict[str, float]


@dataclass(frozen=True)
class BaselineResult:
    """Trained baseline and holdout evidence."""

    model: Pipeline
    metrics: MetricMap
    train_rows: int
    test_rows: int
    test_fraction: float
    train_max_time: float
    holdout_min_time: float
    holdout_probabilities: tuple[float, ...]
    holdout_labels: tuple[int, ...]


@dataclass(frozen=True)
class ChronologicalSplit:
    """Leakage-safe chronological train/holdout partition."""

    train: pd.DataFrame
    holdout: pd.DataFrame
    test_fraction: float

    @property
    def train_max_time(self) -> float:
        """Latest timestamp made available to model training."""
        return float(self.train["Time"].max())

    @property
    def holdout_min_time(self) -> float:
        """Earliest timestamp reserved for holdout evaluation."""
        return float(self.holdout["Time"].min())

    def to_report(self) -> dict[str, float | int | str]:
        """Return JSON-serializable split evidence."""
        return {
            "strategy": "chronological",
            "train_rows": len(self.train),
            "test_rows": len(self.holdout),
            "test_fraction": self.test_fraction,
            "train_max_time": self.train_max_time,
            "holdout_min_time": self.holdout_min_time,
        }


def chronological_train_holdout(
    frame: pd.DataFrame, *, test_fraction: float = 0.2
) -> ChronologicalSplit:
    """Validate transactions and split newest observations into the holdout set."""
    validate_transactions(frame)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    split_index = int(len(ordered) * (1.0 - test_fraction))
    if split_index == 0 or split_index == len(ordered):
        raise ValueError("dataset is too small for the requested holdout")
    return ChronologicalSplit(
        train=ordered.iloc[:split_index],
        holdout=ordered.iloc[split_index:],
        test_fraction=test_fraction,
    )


def train_baseline(frame: pd.DataFrame, *, test_fraction: float = 0.2) -> BaselineResult:
    """Train a class-weighted logistic baseline using a chronological holdout."""
    split = chronological_train_holdout(frame, test_fraction=test_fraction)
    train = split.train
    test = split.holdout
    if train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
        raise ValueError("both chronological partitions must contain fraud and genuine rows")

    preprocessing = ColumnTransformer(
        [("scaled", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42),
            ),
        ]
    )

    train_x = train.loc[:, FEATURE_COLUMNS]
    train_y = train[TARGET_COLUMN]
    test_x = test.loc[:, FEATURE_COLUMNS]
    test_y = test[TARGET_COLUMN]
    model.fit(train_x, train_y)

    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: MetricMap = {
        "average_precision": float(average_precision_score(test_y, probabilities)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "precision": float(precision_score(test_y, predictions, zero_division=0)),
        "recall": float(recall_score(test_y, predictions, zero_division=0)),
        "f1": float(f1_score(test_y, predictions, zero_division=0)),
    }
    return BaselineResult(
        model=model,
        metrics=metrics,
        train_rows=len(train),
        test_rows=len(test),
        test_fraction=test_fraction,
        train_max_time=split.train_max_time,
        holdout_min_time=split.holdout_min_time,
        holdout_probabilities=tuple(float(value) for value in probabilities.tolist()),
        holdout_labels=tuple(int(value) for value in test_y.tolist()),
    )
