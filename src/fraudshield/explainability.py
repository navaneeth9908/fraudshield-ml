"""Model-agnostic explanations and slice-based error analysis."""

from __future__ import annotations

from math import isinf
from typing import cast

import numpy as np
import pandas as pd

from fraudshield.data import FEATURE_COLUMNS, TARGET_COLUMN, validate_transactions
from fraudshield.modeling import CandidateEvaluation, compare_model_candidates

AMOUNT_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 25.0, "amount:[0, 25)"),
    (25.0, 100.0, "amount:[25, 100)"),
    (100.0, 500.0, "amount:[100, 500)"),
    (500.0, float("inf"), "amount:[500, inf)"),
)
TIME_WINDOW_NAMES: tuple[str, str, str] = (
    "time:early_holdout",
    "time:middle_holdout",
    "time:late_holdout",
)
TOP_TRANSACTION_CONTRIBUTIONS = 5


def amount_band_for_value(amount: float) -> str:
    """Return the deterministic amount-band label for one transaction amount."""
    value = float(amount)
    for lower, upper, label in AMOUNT_BANDS:
        if value >= lower and (value < upper or isinf(upper)):
            return label
    return AMOUNT_BANDS[0][2]


def generate_explainability_report(
    frame: pd.DataFrame,
    *,
    explanation_sample_size: int = 200,
    holdout_fraction: float = 0.2,
    calibration_fraction: float = 0.2,
    false_negative_cost: float = 500.0,
    false_positive_cost: float = 25.0,
    analyst_alert_budget: int = 1_000,
) -> dict[str, object]:
    """Train the selected fraud model and return versioned explanation evidence.

    Explanations use a deterministic model-agnostic perturbation fallback: each sampled
    holdout transaction is rescored after replacing one feature at a time with the
    model-train median for that feature. Positive deltas indicate that the observed
    feature value raised the calibrated fraud probability relative to that baseline.
    """
    if explanation_sample_size < 1:
        raise ValueError("explanation_sample_size must be at least 1")
    validate_transactions(frame)
    comparison = compare_model_candidates(
        frame,
        holdout_fraction=holdout_fraction,
        calibration_fraction=calibration_fraction,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        analyst_alert_budget=analyst_alert_budget,
    )
    selected = comparison.evaluation_by_name(comparison.selected_model)
    holdout = comparison.split.holdout.reset_index(drop=True)
    probabilities = np.asarray(selected.holdout_probabilities, dtype=float)
    labels = np.asarray(selected.holdout_labels, dtype=int)
    threshold = float(selected.decision.chosen_threshold)
    predictions = probabilities >= threshold

    sample_indexes = _representative_sample_indexes(probabilities, explanation_sample_size)
    sample_features = (
        holdout.iloc[list(sample_indexes)].loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
    )
    baseline = comparison.split.model_train.loc[:, list(FEATURE_COLUMNS)].median(numeric_only=True)
    baseline = baseline.reindex(list(FEATURE_COLUMNS))
    contribution_matrix, sample_probabilities = _perturbation_contributions(
        selected, sample_features, baseline
    )

    global_importance = _global_importance(contribution_matrix)
    transaction_explanations = _transaction_explanations(
        sample_indexes=sample_indexes,
        sample_probabilities=sample_probabilities,
        sample_labels=labels[list(sample_indexes)],
        threshold=threshold,
        contribution_matrix=contribution_matrix,
    )
    split_report = comparison.split.to_report()
    decision_report = _decision_report(selected)
    return {
        "schema_version": 1,
        "selected_model": comparison.selected_model,
        "model_config": selected.model_config,
        "selection_rule": list(comparison.selection_rule),
        "split": split_report,
        "decision": decision_report,
        "feature_metadata": {
            "input_features": list(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "target_column_excluded_from_model_inputs": True,
            "raw_features": ["Time", "Amount"],
            "pca_anonymized_features": [f"V{index}" for index in range(1, 29)],
            "pca_anonymized_feature_count": 28,
            "limitations": [
                "V1-V28 are anonymized PCA components, so business-causal feature names "
                "are unavailable.",
                "Perturbation deltas are local association evidence, not proof of causal drivers.",
                "Slice error analysis uses labeled holdout data and should not be used "
                "as model input.",
            ],
        },
        "leakage_controls": {
            "model_input_columns": list(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "target_available_only_for_holdout_evaluation": True,
            "explanation_baseline_source": "model_train_feature_medians",
            "explanation_baseline_columns": list(FEATURE_COLUMNS),
            "model_fit_max_time": split_report["model_train_max_time"],
            "calibration_max_time": split_report["calibration_max_time"],
            "holdout_min_time": split_report["holdout_min_time"],
        },
        "explanation": {
            "method": "model_agnostic_feature_perturbation",
            "baseline": {
                "source": "model_train_feature_medians",
                "feature_count": len(FEATURE_COLUMNS),
                "target_included": False,
            },
            "sample": {
                "strategy": "deterministic_probability_quantiles_from_holdout",
                "requested_rows": explanation_sample_size,
                "actual_rows": len(sample_indexes),
                "candidate_rows": len(holdout),
                "model_prediction_calls": len(sample_indexes) * (len(FEATURE_COLUMNS) + 1),
            },
        },
        "global_importance": global_importance,
        "transaction_explanations": transaction_explanations,
        "error_analysis": _error_analysis(
            holdout=holdout,
            labels=labels,
            probabilities=probabilities,
            predictions=predictions,
            threshold=threshold,
        ),
    }


def render_model_card(report: dict[str, object]) -> str:
    """Render a concise Markdown model card from an explainability report."""
    selected_model = cast(str, report["selected_model"])
    split = cast(dict[str, object], report["split"])
    decision = cast(dict[str, object], report["decision"])
    explanation = cast(dict[str, object], report["explanation"])
    sample = cast(dict[str, object], explanation["sample"])
    feature_metadata = cast(dict[str, object], report["feature_metadata"])
    global_importance = cast(list[dict[str, object]], report["global_importance"])
    error_analysis = cast(dict[str, object], report["error_analysis"])
    amount_bands = cast(list[dict[str, object]], error_analysis["amount_bands"])
    time_windows = cast(list[dict[str, object]], error_analysis["time_windows"])

    top_features = "\n".join(
        f"| {entry['rank']} | `{entry['feature']}` | "
        f"{cast(float, entry['mean_absolute_probability_delta']):.6f} | "
        f"{cast(float, entry['mean_probability_delta']):.6f} |"
        for entry in global_importance[:10]
    )
    amount_rows = "\n".join(_slice_markdown_row(group) for group in amount_bands)
    time_rows = "\n".join(_slice_markdown_row(group) for group in time_windows)
    pca_features = cast(list[str], feature_metadata["pca_anonymized_features"])
    split_summary = (
        f"- Chronological split: {split['model_train_rows']} fit rows, "
        f"{split['calibration_rows']} calibration rows, "
        f"{split['holdout_rows']} final holdout rows"
    )
    threshold_summary = (
        f"- Operating threshold: {cast(float, decision['chosen_threshold']):.6f} "
        "from minimum expected-loss analysis"
    )
    intended_use = (
        "FraudShield is an educational portfolio fraud-risk scoring system for offline "
        "analysis of the Kaggle credit-card fraud dataset. It demonstrates leakage-safe "
        "model comparison, calibrated probabilities, cost-sensitive thresholding, "
        "model-agnostic explanations, and holdout slice diagnostics. It is not a live "
        "payment authorization system."
    )
    feature_context = (
        f"The model uses `Time`, `Amount`, and anonymized PCA components `{pca_features[0]}` "
        f"through `{pca_features[-1]}`. The `Class` target is excluded from model inputs "
        "and appears only in holdout evaluation/error-analysis aggregates. Because the "
        "PCA components are anonymized, explanations can rank component influence but "
        "cannot map those components back to customer, merchant, geography, or device "
        "concepts."
    )
    explanation_summary = (
        f"Explanations use `{explanation['method']}` on a deterministic holdout sample of "
        f"{sample['actual_rows']} rows from {sample['candidate_rows']} candidates. Each "
        "sampled row is rescored after replacing one feature with its model-train median. "
        "Positive deltas mean the observed value increased calibrated fraud probability "
        "relative to that model-train baseline; negative deltas mean it lowered risk."
    )
    wide_table_header = (
        "| Slice | Rows | Fraud cases | Alerts | False positives | False negatives | "
        "Precision | Recall | Fraud rate |"
    )
    return "\n".join(
        [
            "# FraudShield model card",
            "",
            "## Model summary",
            "",
            f"- Selected model: `{selected_model}`",
            "- Selection rule: PR-AUC first, then expected loss, recall, precision, "
            "and stable model-name tie break",
            split_summary,
            threshold_summary,
            "",
            "## Intended use",
            "",
            intended_use,
            "",
            "## Features and privacy context",
            "",
            feature_context,
            "",
            "## Explanation method",
            "",
            explanation_summary,
            "",
            "| Rank | Feature | Mean absolute probability delta | Mean signed probability delta |",
            "|---:|---|---:|---:|",
            top_features,
            "",
            "## Slice-based error analysis",
            "",
            "Amount bands:",
            "",
            wide_table_header,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            amount_rows,
            "",
            "Time windows:",
            "",
            wide_table_header,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            time_rows,
            "",
            "## Limitations and responsible use",
            "",
            "- The data represents two days of European cardholder transactions from "
            "September 2013 and is not current production traffic.",
            "- The PCA-anonymized `V1`-`V28` features limit human interpretability and "
            "fairness analysis.",
            "- Perturbation explanations are practical, deterministic model-agnostic "
            "diagnostics, not causal claims.",
            "- Error-analysis slices use labeled holdout data and should remain evaluation "
            "evidence, not features for fitting or calibration.",
            "- A production deployment would require current labels, drift monitoring, review "
            "workflows, security controls, governance, and fairness evaluation.",
            "",
        ]
    )


def _representative_sample_indexes(
    probabilities: np.ndarray, requested_rows: int
) -> tuple[int, ...]:
    sample_size = min(requested_rows, len(probabilities))
    ranked = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    if sample_size == len(ranked):
        return tuple(ranked)
    positions = np.linspace(0, len(ranked) - 1, num=sample_size)
    return tuple(ranked[round(position)] for position in positions)


def _perturbation_contributions(
    selected: CandidateEvaluation, sample_features: pd.DataFrame, baseline: pd.Series
) -> tuple[dict[str, tuple[float, ...]], tuple[float, ...]]:
    base_probabilities = selected.model.predict_proba(sample_features)[:, 1]
    contributions: dict[str, tuple[float, ...]] = {}
    for feature in FEATURE_COLUMNS:
        perturbed = sample_features.copy()
        perturbed[feature] = baseline[feature]
        perturbed_probabilities = selected.model.predict_proba(perturbed)[:, 1]
        deltas = base_probabilities - perturbed_probabilities
        contributions[feature] = tuple(_round_float(value) for value in deltas.tolist())
    return contributions, tuple(_round_float(value) for value in base_probabilities.tolist())


def _global_importance(
    contribution_matrix: dict[str, tuple[float, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature, values in contribution_matrix.items():
        array = np.asarray(values, dtype=float)
        mean_delta = _round_float(float(np.mean(array)))
        rows.append(
            {
                "feature": feature,
                "mean_probability_delta": mean_delta,
                "mean_absolute_probability_delta": _round_float(float(np.mean(np.abs(array)))),
                "direction": _direction(mean_delta),
            }
        )
    rows.sort(
        key=lambda entry: (
            -cast(float, entry["mean_absolute_probability_delta"]),
            cast(str, entry["feature"]),
        )
    )
    return [{"rank": rank, **entry} for rank, entry in enumerate(rows, start=1)]


def _transaction_explanations(
    *,
    sample_indexes: tuple[int, ...],
    sample_probabilities: tuple[float, ...],
    sample_labels: np.ndarray,
    threshold: float,
    contribution_matrix: dict[str, tuple[float, ...]],
) -> list[dict[str, object]]:
    explanations: list[dict[str, object]] = []
    for sample_position, holdout_index in enumerate(sample_indexes):
        probability = sample_probabilities[sample_position]
        predicted_label = int(probability >= threshold)
        actual_label = int(sample_labels[sample_position])
        contributions = [
            {
                "feature": feature,
                "probability_delta": contribution_matrix[feature][sample_position],
                "direction": _direction(contribution_matrix[feature][sample_position]),
            }
            for feature in FEATURE_COLUMNS
        ]
        contributions.sort(
            key=lambda contribution: (
                -abs(cast(float, contribution["probability_delta"])),
                cast(str, contribution["feature"]),
            )
        )
        explanations.append(
            {
                "rank": sample_position + 1,
                "holdout_row_index": int(holdout_index),
                "fraud_probability": probability,
                "predicted_label": predicted_label,
                "actual_label": actual_label,
                "error_type": _error_type(
                    actual_label=actual_label, predicted_label=predicted_label
                ),
                "top_contributions": contributions[:TOP_TRANSACTION_CONTRIBUTIONS],
            }
        )
    return explanations


def _error_analysis(
    *,
    holdout: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    amount_labels = [amount_band_for_value(float(amount)) for amount in holdout["Amount"]]
    amount_bands = [
        _slice_metrics(
            name=band_label,
            mask=np.asarray([label == band_label for label in amount_labels], dtype=bool),
            labels=labels,
            predictions=predictions,
        )
        for _, _, band_label in AMOUNT_BANDS
    ]
    time_windows = [
        _slice_metrics(name=name, mask=mask, labels=labels, predictions=predictions)
        for name, mask in _time_window_masks(len(holdout))
    ]
    return {
        "threshold_source": "selected_model_minimum_expected_loss",
        "threshold": _round_float(threshold),
        "amount_bands": amount_bands,
        "time_windows": time_windows,
        "probability_summary": {
            "min": _round_float(float(np.min(probabilities))),
            "max": _round_float(float(np.max(probabilities))),
            "mean": _round_float(float(np.mean(probabilities))),
        },
    }


def _time_window_masks(row_count: int) -> tuple[tuple[str, np.ndarray], ...]:
    indexes = np.arange(row_count)
    boundaries = [round(row_count * fraction / 3.0) for fraction in range(4)]
    boundaries[0] = 0
    boundaries[-1] = row_count
    return tuple(
        (
            TIME_WINDOW_NAMES[position],
            (indexes >= boundaries[position]) & (indexes < boundaries[position + 1]),
        )
        for position in range(3)
    )


def _slice_metrics(
    *, name: str, mask: np.ndarray, labels: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    slice_labels = labels[mask]
    slice_predictions = predictions[mask]
    rows = int(mask.sum())
    fraud_cases = int(slice_labels.sum())
    alerts = int(slice_predictions.sum())
    true_positives = int(((slice_labels == 1) & slice_predictions).sum())
    false_positives = int(((slice_labels == 0) & slice_predictions).sum())
    false_negatives = int(((slice_labels == 1) & ~slice_predictions).sum())
    precision = true_positives / alerts if alerts else 0.0
    recall = true_positives / fraud_cases if fraud_cases else 0.0
    fraud_rate = fraud_cases / rows if rows else 0.0
    return {
        "name": name,
        "rows": rows,
        "fraud_cases": fraud_cases,
        "alerts": alerts,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": _round_float(precision),
        "recall": _round_float(recall),
        "fraud_rate": _round_float(fraud_rate),
    }


def _decision_report(selected: CandidateEvaluation) -> dict[str, object]:
    decision = selected.decision
    return {
        "chosen_threshold": _round_float(decision.chosen_threshold),
        "expected_loss": _round_float(decision.expected_loss),
        "precision": _round_float(decision.precision),
        "recall": _round_float(decision.recall),
        "f1": _round_float(decision.f1),
        "alert_count": decision.alert_count,
        "thresholds_evaluated": len(decision.threshold_sweep),
        "analyst_budget": {
            "alert_count": decision.budget_alert_count,
            "precision": _round_float(decision.budget_precision),
            "recall": _round_float(decision.budget_recall),
        },
    }


def _error_type(*, actual_label: int, predicted_label: int) -> str:
    if actual_label == 1 and predicted_label == 1:
        return "true_positive"
    if actual_label == 0 and predicted_label == 1:
        return "false_positive"
    if actual_label == 1 and predicted_label == 0:
        return "false_negative"
    return "true_negative"


def _direction(delta: float) -> str:
    if delta > 0.0:
        return "increases_risk"
    if delta < 0.0:
        return "decreases_risk"
    return "neutral"


def _slice_markdown_row(group: dict[str, object]) -> str:
    return (
        f"| `{group['name']}` | {group['rows']} | {group['fraud_cases']} | {group['alerts']} | "
        f"{group['false_positives']} | {group['false_negatives']} | "
        f"{cast(float, group['precision']):.4f} | {cast(float, group['recall']):.4f} | "
        f"{cast(float, group['fraud_rate']):.4f} |"
    )


def _round_float(value: float) -> float:
    return float(round(value, 12))
