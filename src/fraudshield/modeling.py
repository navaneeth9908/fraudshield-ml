"""Model training, calibration, and leakage-safe chronological evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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
from fraudshield.decision import DecisionCosts, DecisionResult, optimize_decisions

MetricMap = dict[str, float]
SELECTION_RULE: tuple[str, ...] = (
    "maximize_average_precision",
    "minimize_expected_loss",
    "maximize_recall",
    "maximize_precision",
    "lexicographic_model_name",
)


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


@dataclass(frozen=True)
class ChronologicalModelingSplit:
    """Nested chronological split for fitting, calibration, and final holdout scoring."""

    model_train: pd.DataFrame
    calibration: pd.DataFrame
    holdout: pd.DataFrame
    holdout_fraction: float
    calibration_fraction: float

    @property
    def model_train_max_time(self) -> float:
        """Latest timestamp used to fit base estimators and preprocessing."""
        return float(self.model_train["Time"].max())

    @property
    def calibration_max_time(self) -> float:
        """Latest timestamp used for probability calibration."""
        return float(self.calibration["Time"].max())

    @property
    def holdout_min_time(self) -> float:
        """Earliest timestamp reserved for final comparison evidence."""
        return float(self.holdout["Time"].min())

    def to_report(self) -> dict[str, float | int | str]:
        """Return JSON-serializable leakage-boundary evidence."""
        return {
            "strategy": "chronological_model_calibration_holdout",
            "model_train_rows": len(self.model_train),
            "calibration_rows": len(self.calibration),
            "holdout_rows": len(self.holdout),
            "holdout_fraction": self.holdout_fraction,
            "calibration_fraction_of_pre_holdout": self.calibration_fraction,
            "model_train_max_time": self.model_train_max_time,
            "calibration_min_time": float(self.calibration["Time"].min()),
            "calibration_max_time": self.calibration_max_time,
            "holdout_min_time": self.holdout_min_time,
        }


@dataclass(frozen=True)
class CalibratedFraudModel:
    """A fitted preprocessing/classifier pipeline wrapped with a calibration layer."""

    base_model: Pipeline
    calibrator: LogisticRegression

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        """Return calibrated fraud-class probabilities."""
        raw_probabilities = self.base_model.predict_proba(frame)[:, 1]
        calibrated_fraud = self.calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
        return np.column_stack((1.0 - calibrated_fraud, calibrated_fraud))


@dataclass(frozen=True)
class CandidateEvaluation:
    """Holdout evaluation for one calibrated candidate model."""

    name: str
    model: CalibratedFraudModel
    metrics: MetricMap
    decision: DecisionResult
    model_config: dict[str, object]
    threshold_config: dict[str, object]
    calibration_method: str
    fit_rows: int
    calibration_rows: int
    holdout_rows: int
    fit_max_time: float
    calibration_max_time: float
    holdout_min_time: float
    holdout_probabilities: tuple[float, ...]
    holdout_labels: tuple[int, ...]

    def to_report(self) -> dict[str, object]:
        """Return compact machine-readable evidence without serializing model binaries."""
        return {
            "name": self.name,
            "model_config": self.model_config,
            "metrics": self.metrics,
            "decision": _compact_decision_report(self.decision),
            "threshold": self.threshold_config,
            "calibration": {
                "method": self.calibration_method,
                "fit_on": "calibration",
                "holdout_used": False,
                "rows": self.calibration_rows,
                "max_time": self.calibration_max_time,
            },
            "fit": {"rows": self.fit_rows, "max_time": self.fit_max_time},
            "holdout": {"rows": self.holdout_rows, "min_time": self.holdout_min_time},
            "holdout_probability_count": len(self.holdout_probabilities),
            "holdout_probability_summary": {
                "min": float(min(self.holdout_probabilities)),
                "max": float(max(self.holdout_probabilities)),
                "mean": float(np.mean(self.holdout_probabilities)),
            },
        }


@dataclass(frozen=True)
class ModelComparisonResult:
    """Leakage-safe comparison of calibrated fraud-risk model candidates."""

    split: ChronologicalModelingSplit
    candidates: tuple[CandidateEvaluation, ...]
    selected_model: str
    selection_rule: tuple[str, ...]
    costs: DecisionCosts
    analyst_alert_budget: int

    def evaluation_by_name(self, name: str) -> CandidateEvaluation:
        """Return one candidate evaluation by stable candidate name."""
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"Unknown candidate model: {name}")

    def to_report(self) -> dict[str, object]:
        """Return versioned machine-readable comparison evidence."""
        return {
            "schema_version": 1,
            "selection_rule": list(self.selection_rule),
            "selected_model": self.selected_model,
            "costs": {
                "false_negative": self.costs.false_negative,
                "false_positive": self.costs.false_positive,
            },
            "analyst_alert_budget": self.analyst_alert_budget,
            "split": self.split.to_report(),
            "candidates": [candidate.to_report() for candidate in self.candidates],
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


def chronological_modeling_split(
    frame: pd.DataFrame, *, holdout_fraction: float = 0.2, calibration_fraction: float = 0.2
) -> ChronologicalModelingSplit:
    """Split transactions into base-fit, calibration, and untouched holdout windows."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    outer_split = chronological_train_holdout(frame, test_fraction=holdout_fraction)
    pre_holdout = outer_split.train
    calibration_rows = int(len(pre_holdout) * calibration_fraction)
    model_train_rows = len(pre_holdout) - calibration_rows
    if model_train_rows == 0 or calibration_rows == 0:
        raise ValueError("dataset is too small for the requested calibration split")
    return ChronologicalModelingSplit(
        model_train=pre_holdout.iloc[:model_train_rows],
        calibration=pre_holdout.iloc[model_train_rows:],
        holdout=outer_split.holdout,
        holdout_fraction=holdout_fraction,
        calibration_fraction=calibration_fraction,
    )


def train_baseline(frame: pd.DataFrame, *, test_fraction: float = 0.2) -> BaselineResult:
    """Train a class-weighted logistic baseline using a chronological holdout."""
    split = chronological_train_holdout(frame, test_fraction=test_fraction)
    train = split.train
    test = split.holdout
    if train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
        raise ValueError("both chronological partitions must contain fraud and genuine rows")

    model = Pipeline(
        [
            ("preprocessing", _preprocessing()),
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


def compare_model_candidates(
    frame: pd.DataFrame,
    *,
    holdout_fraction: float = 0.2,
    calibration_fraction: float = 0.2,
    false_negative_cost: float = 500.0,
    false_positive_cost: float = 25.0,
    analyst_alert_budget: int = 1_000,
    calibration_method: str = "sigmoid",
) -> ModelComparisonResult:
    """Train, calibrate, and compare logistic and gradient-boosting fraud candidates."""
    split = chronological_modeling_split(
        frame, holdout_fraction=holdout_fraction, calibration_fraction=calibration_fraction
    )
    _require_binary_partitions(split)
    costs = DecisionCosts(false_negative=false_negative_cost, false_positive=false_positive_cost)
    if analyst_alert_budget < 1 or analyst_alert_budget > len(split.holdout):
        raise ValueError("analyst_alert_budget must be between 1 and holdout row count")

    candidates = tuple(
        _evaluate_candidate(
            name=name,
            classifier=classifier,
            config=config,
            split=split,
            costs=costs,
            analyst_alert_budget=analyst_alert_budget,
            calibration_method=calibration_method,
        )
        for name, classifier, config in _candidate_specs()
    )
    selected = max(candidates, key=_selection_key)
    return ModelComparisonResult(
        split=split,
        candidates=candidates,
        selected_model=selected.name,
        selection_rule=SELECTION_RULE,
        costs=costs,
        analyst_alert_budget=analyst_alert_budget,
    )


def _preprocessing() -> ColumnTransformer:
    return ColumnTransformer(
        [("scaled", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
    )


def _candidate_specs() -> tuple[tuple[str, object, dict[str, object]], ...]:
    return (
        (
            "logistic_regression",
            LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42),
            {
                "estimator": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1_000,
                "random_state": 42,
            },
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                class_weight="balanced",
                early_stopping=False,
                l2_regularization=0.01,
                learning_rate=0.08,
                max_iter=80,
                max_leaf_nodes=15,
                random_state=42,
            ),
            {
                "estimator": "HistGradientBoostingClassifier",
                "class_weight": "balanced",
                "early_stopping": False,
                "l2_regularization": 0.01,
                "learning_rate": 0.08,
                "max_iter": 80,
                "max_leaf_nodes": 15,
                "random_state": 42,
            },
        ),
    )


def _evaluate_candidate(
    *,
    name: str,
    classifier: object,
    config: dict[str, object],
    split: ChronologicalModelingSplit,
    costs: DecisionCosts,
    analyst_alert_budget: int,
    calibration_method: str,
) -> CandidateEvaluation:
    base_model = Pipeline([("preprocessing", _preprocessing()), ("classifier", classifier)])
    train_x = split.model_train.loc[:, list(FEATURE_COLUMNS)]
    train_y = split.model_train[TARGET_COLUMN]
    calibration_x = split.calibration.loc[:, list(FEATURE_COLUMNS)]
    calibration_y = split.calibration[TARGET_COLUMN]
    holdout_x = split.holdout.loc[:, list(FEATURE_COLUMNS)]
    holdout_y = split.holdout[TARGET_COLUMN]

    base_model.fit(train_x, train_y)
    if calibration_method != "sigmoid":
        raise ValueError("only sigmoid calibration is currently supported")
    calibration_scores = base_model.predict_proba(calibration_x)[:, 1]
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=1_000, random_state=42)
    calibrator.fit(calibration_scores.reshape(-1, 1), calibration_y)
    calibrated_model = CalibratedFraudModel(base_model=base_model, calibrator=calibrator)
    probabilities = calibrated_model.predict_proba(holdout_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    labels = tuple(int(value) for value in holdout_y.tolist())
    probability_tuple = tuple(float(value) for value in probabilities.tolist())
    metrics: MetricMap = {
        "average_precision": float(average_precision_score(holdout_y, probabilities)),
        "roc_auc": float(roc_auc_score(holdout_y, probabilities)),
        "precision_at_0_50": float(precision_score(holdout_y, predictions, zero_division=0)),
        "recall_at_0_50": float(recall_score(holdout_y, predictions, zero_division=0)),
        "f1_at_0_50": float(f1_score(holdout_y, predictions, zero_division=0)),
    }
    decision = optimize_decisions(
        probabilities=probability_tuple,
        labels=labels,
        costs=costs,
        analyst_alert_budget=analyst_alert_budget,
    )
    return CandidateEvaluation(
        name=name,
        model=calibrated_model,
        metrics=metrics,
        decision=decision,
        model_config=config,
        threshold_config={
            "selected_on": "holdout_probabilities",
            "method": "minimum_expected_loss",
            "false_negative_cost": costs.false_negative,
            "false_positive_cost": costs.false_positive,
            "analyst_alert_budget": analyst_alert_budget,
        },
        calibration_method=calibration_method,
        fit_rows=len(split.model_train),
        calibration_rows=len(split.calibration),
        holdout_rows=len(split.holdout),
        fit_max_time=split.model_train_max_time,
        calibration_max_time=split.calibration_max_time,
        holdout_min_time=split.holdout_min_time,
        holdout_probabilities=probability_tuple,
        holdout_labels=labels,
    )


def _require_binary_partitions(split: ChronologicalModelingSplit) -> None:
    for name, partition in (
        ("model_train", split.model_train),
        ("calibration", split.calibration),
        ("holdout", split.holdout),
    ):
        if partition[TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"{name} partition must contain fraud and genuine rows")


def _selection_key(candidate: CandidateEvaluation) -> tuple[float, float, float, float, str]:
    return (
        candidate.metrics["average_precision"],
        -candidate.decision.expected_loss,
        candidate.decision.recall,
        candidate.decision.precision,
        _reverse_lexicographic_key(candidate.name),
    )


def _reverse_lexicographic_key(value: str) -> str:
    return "".join(chr(0x10FFFF - ord(character)) for character in value)


def _compact_decision_report(decision: DecisionResult) -> dict[str, object]:
    return {
        "chosen_threshold": decision.chosen_threshold,
        "expected_loss": decision.expected_loss,
        "precision": decision.precision,
        "recall": decision.recall,
        "f1": decision.f1,
        "alert_count": decision.alert_count,
        "thresholds_evaluated": len(decision.threshold_sweep),
        "analyst_budget": {
            "alert_count": decision.budget_alert_count,
            "precision": decision.budget_precision,
            "recall": decision.budget_recall,
        },
    }
