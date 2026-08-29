from __future__ import annotations

import numpy as np
import pandas as pd


def _training_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    fraud = np.zeros(rows, dtype=int)
    fraud[::20] = 1
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{f"V{index}": rng.normal(size=rows) for index in range(1, 29)},
            "Amount": rng.lognormal(mean=3.0, sigma=1.0, size=rows),
            "Class": fraud,
        }
    )


def _comparison_frame(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    fraud = (np.arange(rows) % 15 == 0).astype(int)
    signal = fraud * 3.0 + rng.normal(scale=0.35, size=rows)
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            "V1": signal,
            **{f"V{index}": rng.normal(size=rows) for index in range(2, 29)},
            "Amount": 20.0 + fraud * 80.0 + rng.normal(scale=2.0, size=rows),
            "Class": fraud,
        }
    )


def test_baseline_training_returns_leakage_safe_holdout_metrics() -> None:
    from fraudshield.modeling import train_baseline

    result = train_baseline(_training_frame(), test_fraction=0.25)

    assert result.train_rows == 180
    assert result.test_rows == 60
    assert set(result.metrics) == {
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    }
    assert all(0.0 <= value <= 1.0 for value in result.metrics.values())
    assert result.model.classes_.tolist() == [0, 1]


def test_baseline_exposes_chronological_holdout_probabilities_without_target_leakage() -> None:
    from fraudshield.modeling import train_baseline

    frame = _training_frame()
    result = train_baseline(frame, test_fraction=0.25)

    assert len(result.holdout_probabilities) == result.test_rows
    assert len(result.holdout_labels) == result.test_rows
    assert all(0.0 <= probability <= 1.0 for probability in result.holdout_probabilities)
    assert result.holdout_labels == tuple(frame.sort_values("Time").tail(60)["Class"].astype(int))


def test_advanced_split_keeps_model_calibration_and_holdout_chronological() -> None:
    from fraudshield.modeling import chronological_modeling_split

    split = chronological_modeling_split(
        _comparison_frame(), holdout_fraction=0.2, calibration_fraction=0.2
    )

    assert len(split.model_train) == 192
    assert len(split.calibration) == 48
    assert len(split.holdout) == 60
    assert split.model_train["Time"].max() < split.calibration["Time"].min()
    assert split.calibration["Time"].max() < split.holdout["Time"].min()
    assert split.to_report()["strategy"] == "chronological_model_calibration_holdout"


def test_model_comparison_calibrates_without_holdout_fitted_preprocessing() -> None:
    from fraudshield.modeling import compare_model_candidates

    frame = _comparison_frame()
    result = compare_model_candidates(
        frame,
        holdout_fraction=0.2,
        calibration_fraction=0.2,
        analyst_alert_budget=20,
    )
    logistic = result.evaluation_by_name("logistic_regression")
    scaler = logistic.model.base_model.named_steps["preprocessing"].named_transformers_["scaled"]
    model_train_time_mean = result.split.model_train["Time"].mean()

    assert scaler.mean_[0] == model_train_time_mean
    assert logistic.calibration_rows == len(result.split.calibration)
    assert logistic.fit_max_time == float(result.split.model_train["Time"].max())
    assert logistic.calibration_max_time < float(result.split.holdout["Time"].min())
    assert len(logistic.holdout_probabilities) == len(result.split.holdout)
    assert all(0.0 <= score <= 1.0 for score in logistic.holdout_probabilities)


def test_model_comparison_is_repeatable_and_selects_by_pr_auc_then_expected_loss() -> None:
    from fraudshield.modeling import compare_model_candidates

    frame = _comparison_frame()
    first = compare_model_candidates(frame, analyst_alert_budget=20)
    second = compare_model_candidates(frame, analyst_alert_budget=20)

    assert first.to_report() == second.to_report()
    selected = first.evaluation_by_name(first.selected_model)
    for candidate in first.candidates:
        assert (
            selected.metrics["average_precision"],
            -selected.decision.expected_loss,
            selected.decision.recall,
        ) >= (
            candidate.metrics["average_precision"],
            -candidate.decision.expected_loss,
            candidate.decision.recall,
        )
    assert first.to_report()["schema_version"] == 1
    assert first.to_report()["selection_rule"] == [
        "maximize_average_precision",
        "minimize_expected_loss",
        "maximize_recall",
        "maximize_precision",
        "lexicographic_model_name",
    ]
    assert {candidate["name"] for candidate in first.to_report()["candidates"]} == {
        "logistic_regression",
        "hist_gradient_boosting",
    }
