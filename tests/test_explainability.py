from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.data import FEATURE_COLUMNS


def _explainability_frame(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    fraud = (np.arange(rows) % 18 == 0).astype(int)
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            "V1": fraud * 3.0 + rng.normal(scale=0.25, size=rows),
            "V2": -fraud * 2.0 + rng.normal(scale=0.30, size=rows),
            **{f"V{index}": rng.normal(size=rows) for index in range(3, 29)},
            "Amount": 15.0 + fraud * 160.0 + (np.arange(rows) % 7) * 18.0,
            "Class": fraud,
        }
    )


def test_explainability_report_schema_stable_sampling_and_no_target_leakage() -> None:
    from fraudshield.explainability import generate_explainability_report

    frame = _explainability_frame()
    first = generate_explainability_report(
        frame,
        explanation_sample_size=8,
        analyst_alert_budget=20,
    )
    second = generate_explainability_report(
        frame,
        explanation_sample_size=8,
        analyst_alert_budget=20,
    )

    assert first == second
    assert first["schema_version"] == 1
    assert first["selected_model"] in {"logistic_regression", "hist_gradient_boosting"}
    assert first["explanation"]["method"] == "model_agnostic_feature_perturbation"
    assert first["explanation"]["sample"]["requested_rows"] == 8
    assert first["explanation"]["sample"]["actual_rows"] == 8
    assert first["explanation"]["sample"]["candidate_rows"] == first["split"]["holdout_rows"]
    assert first["explanation"]["sample"]["model_prediction_calls"] == 8 * (
        len(FEATURE_COLUMNS) + 1
    )
    assert first["feature_metadata"]["input_features"] == list(FEATURE_COLUMNS)
    assert first["feature_metadata"]["target_column"] == "Class"
    assert first["feature_metadata"]["target_column_excluded_from_model_inputs"] is True
    assert first["feature_metadata"]["pca_anonymized_feature_count"] == 28
    assert "Class" not in first["leakage_controls"]["model_input_columns"]
    assert "Class" not in first["leakage_controls"]["explanation_baseline_columns"]

    importances = first["global_importance"]
    assert {entry["feature"] for entry in importances} == set(FEATURE_COLUMNS)
    assert "Class" not in {entry["feature"] for entry in importances}
    assert importances == sorted(
        importances,
        key=lambda entry: (-entry["mean_absolute_probability_delta"], entry["feature"]),
    )

    transaction_explanations = first["transaction_explanations"]
    assert len(transaction_explanations) == 8
    assert [row["rank"] for row in transaction_explanations] == list(range(1, 9))
    for row in transaction_explanations:
        assert set(row) == {
            "rank",
            "holdout_row_index",
            "fraud_probability",
            "predicted_label",
            "actual_label",
            "error_type",
            "top_contributions",
        }
        assert len(row["top_contributions"]) == 5
        assert "Class" not in {contribution["feature"] for contribution in row["top_contributions"]}
        assert row["top_contributions"] == sorted(
            row["top_contributions"],
            key=lambda contribution: (
                -abs(contribution["probability_delta"]),
                contribution["feature"],
            ),
        )


def test_error_analysis_reports_amount_and_time_slice_boundaries() -> None:
    from fraudshield.explainability import amount_band_for_value, generate_explainability_report

    assert amount_band_for_value(0.0) == "amount:[0, 25)"
    assert amount_band_for_value(24.99) == "amount:[0, 25)"
    assert amount_band_for_value(25.0) == "amount:[25, 100)"
    assert amount_band_for_value(99.99) == "amount:[25, 100)"
    assert amount_band_for_value(100.0) == "amount:[100, 500)"
    assert amount_band_for_value(500.0) == "amount:[500, inf)"

    report = generate_explainability_report(
        _explainability_frame(),
        explanation_sample_size=6,
        analyst_alert_budget=20,
    )
    error_analysis = report["error_analysis"]
    assert error_analysis["threshold_source"] == "selected_model_minimum_expected_loss"
    assert error_analysis["threshold"] == report["decision"]["chosen_threshold"]
    assert {slice_report["name"] for slice_report in error_analysis["amount_bands"]} == {
        "amount:[0, 25)",
        "amount:[25, 100)",
        "amount:[100, 500)",
        "amount:[500, inf)",
    }
    assert [window["name"] for window in error_analysis["time_windows"]] == [
        "time:early_holdout",
        "time:middle_holdout",
        "time:late_holdout",
    ]
    assert (
        sum(slice_report["rows"] for slice_report in error_analysis["amount_bands"])
        == report["split"]["holdout_rows"]
    )
    assert (
        sum(window["rows"] for window in error_analysis["time_windows"])
        == report["split"]["holdout_rows"]
    )
    for group in [*error_analysis["amount_bands"], *error_analysis["time_windows"]]:
        assert set(group) == {
            "name",
            "rows",
            "fraud_cases",
            "alerts",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "fraud_rate",
        }
        assert 0 <= group["alerts"] <= group["rows"]
        assert 0.0 <= group["precision"] <= 1.0
        assert 0.0 <= group["recall"] <= 1.0
        assert 0.0 <= group["fraud_rate"] <= 1.0
