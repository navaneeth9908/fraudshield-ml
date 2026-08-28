"""Cost-sensitive fraud decision metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class DecisionCosts:
    """Business costs for threshold selection."""

    false_negative: float
    false_positive: float

    def __post_init__(self) -> None:
        if self.false_negative <= 0 or not isfinite(self.false_negative):
            raise ValueError("false_negative cost must be a positive finite number")
        if self.false_positive <= 0 or not isfinite(self.false_positive):
            raise ValueError("false_positive cost must be a positive finite number")


@dataclass(frozen=True)
class ThresholdPoint:
    """Metrics observed at one fraud-alert threshold."""

    threshold: float
    expected_loss: float
    precision: float
    recall: float
    f1: float
    alert_count: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass(frozen=True)
class DecisionResult:
    """Chosen threshold, threshold sweep, and analyst-budget evidence."""

    chosen_threshold: float
    expected_loss: float
    precision: float
    recall: float
    f1: float
    alert_count: int
    budget_alert_count: int
    budget_precision: float
    budget_recall: float
    threshold_sweep: tuple[ThresholdPoint, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evidence payload."""
        return {
            "chosen_threshold": self.chosen_threshold,
            "expected_loss": self.expected_loss,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "alert_count": self.alert_count,
            "analyst_budget": {
                "alert_count": self.budget_alert_count,
                "precision": self.budget_precision,
                "recall": self.budget_recall,
            },
            "threshold_sweep": [
                {
                    "threshold": point.threshold,
                    "expected_loss": point.expected_loss,
                    "precision": point.precision,
                    "recall": point.recall,
                    "f1": point.f1,
                    "alert_count": point.alert_count,
                    "false_positives": point.false_positives,
                    "false_negatives": point.false_negatives,
                    "true_positives": point.true_positives,
                }
                for point in self.threshold_sweep
            ],
        }


def optimize_decisions(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    costs: DecisionCosts,
    analyst_alert_budget: int,
) -> DecisionResult:
    """Choose the threshold with minimum fraud decision loss."""
    scores = _validated_probabilities(probabilities)
    actuals = _validated_labels(labels)
    if len(scores) != len(actuals):
        raise ValueError("probabilities and labels must have the same length")
    if analyst_alert_budget < 1 or analyst_alert_budget > len(scores):
        raise ValueError("analyst_alert_budget must be between 1 and transaction count")

    sweep = _threshold_sweep(scores, actuals, costs)
    chosen = min(
        sweep,
        key=lambda point: (
            point.expected_loss,
            -point.recall,
            -point.precision,
            -point.threshold,
        ),
    )
    budget_precision, budget_recall = _budget_metrics(scores, actuals, analyst_alert_budget)
    return DecisionResult(
        chosen_threshold=chosen.threshold,
        expected_loss=chosen.expected_loss,
        precision=chosen.precision,
        recall=chosen.recall,
        f1=chosen.f1,
        alert_count=chosen.alert_count,
        budget_alert_count=analyst_alert_budget,
        budget_precision=budget_precision,
        budget_recall=budget_recall,
        threshold_sweep=sweep,
    )


def _validated_probabilities(probabilities: Sequence[float]) -> tuple[float, ...]:
    if len(probabilities) == 0:
        raise ValueError("probabilities must not be empty")
    scores = tuple(float(score) for score in probabilities)
    if any(not isfinite(score) or score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("probabilities must be finite values between 0 and 1")
    return scores


def _validated_labels(labels: Sequence[int]) -> tuple[int, ...]:
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    actuals = tuple(int(label) for label in labels)
    if any(label not in {0, 1} for label in actuals):
        raise ValueError("labels must contain only 0 and 1")
    return actuals


def _candidate_thresholds(scores: tuple[float, ...]) -> tuple[float, ...]:
    thresholds = {1.0, 0.0, *scores}
    return tuple(sorted(thresholds, reverse=True))


def _threshold_sweep(
    scores: tuple[float, ...],
    actuals: tuple[int, ...],
    costs: DecisionCosts,
) -> tuple[ThresholdPoint, ...]:
    ranked_indexes = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    total_positives = sum(actuals)
    true_positives = 0
    alert_count = 0
    cursor = 0
    points: list[ThresholdPoint] = []
    for threshold in _candidate_thresholds(scores):
        while cursor < len(ranked_indexes) and scores[ranked_indexes[cursor]] >= threshold:
            true_positives += actuals[ranked_indexes[cursor]]
            alert_count += 1
            cursor += 1
        points.append(
            _build_threshold_point(
                threshold=threshold,
                true_positives=true_positives,
                false_positives=alert_count - true_positives,
                false_negatives=total_positives - true_positives,
                alert_count=alert_count,
                costs=costs,
            )
        )
    return tuple(points)


def _build_threshold_point(
    *,
    threshold: float,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    alert_count: int,
    costs: DecisionCosts,
) -> ThresholdPoint:
    expected_loss = false_negatives * costs.false_negative + false_positives * costs.false_positive
    precision, recall, f1 = _classification_metrics(
        true_positives, false_positives, false_negatives
    )
    return ThresholdPoint(
        threshold=threshold,
        expected_loss=expected_loss,
        precision=precision,
        recall=recall,
        f1=f1,
        alert_count=alert_count,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
    )


def _budget_metrics(
    scores: tuple[float, ...], actuals: tuple[int, ...], budget: int
) -> tuple[float, float]:
    ranked_indexes = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    selected = set(ranked_indexes[:budget])
    true_positives = sum(1 for index in selected if actuals[index] == 1)
    false_positives = budget - true_positives
    false_negatives = sum(
        1 for index, actual in enumerate(actuals) if actual == 1 and index not in selected
    )
    precision, recall, _ = _classification_metrics(true_positives, false_positives, false_negatives)
    return precision, recall


def _classification_metrics(
    true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float, float, float]:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
