from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner


def test_population_stability_uses_reference_bins_and_smooths_zero_count_windows() -> None:
    from fraudshield.monitoring import compute_population_stability

    reference = pd.Series(np.arange(10, dtype=float), name="Amount")
    current = pd.Series([0.0, 0.2, 0.4, 0.6, 8.5, 8.7, 8.9, 9.0], name="Amount")

    result = compute_population_stability(reference, current, bins=5, feature_name="Amount")

    assert result.feature == "Amount"
    assert result.bins == 5
    assert result.reference_count == 10
    assert result.current_count == 8
    assert [bucket.lower_bound for bucket in result.buckets] == [0.0, 1.8, 3.6, 5.4, 7.2]
    assert [bucket.upper_bound for bucket in result.buckets] == [1.8, 3.6, 5.4, 7.2, 9.0]
    assert [bucket.current_count for bucket in result.buckets][1:4] == [0, 0, 0]
    assert all(isfinite(bucket.psi_component) for bucket in result.buckets)
    assert isfinite(result.psi)
    assert result.psi > 0.1


def test_monitoring_rejects_empty_nonfinite_and_invalid_threshold_inputs() -> None:
    from fraudshield.monitoring import (
        AlertThresholds,
        compute_population_stability,
        generate_monitoring_report,
    )

    with pytest.raises(ValueError, match="reference window must not be empty"):
        compute_population_stability(
            pd.Series([], dtype=float), pd.Series([1.0]), feature_name="Amount"
        )

    with pytest.raises(ValueError, match="current window must not be empty"):
        compute_population_stability(
            pd.Series([1.0]), pd.Series([], dtype=float), feature_name="Amount"
        )

    with pytest.raises(ValueError, match="finite numeric"):
        compute_population_stability(
            pd.Series([1.0, np.inf]), pd.Series([1.0]), feature_name="Amount"
        )

    with pytest.raises(ValueError, match="warning threshold must be lower than critical threshold"):
        AlertThresholds(warning=0.3, critical=0.2)

    with pytest.raises(ValueError, match="missing required monitoring columns"):
        generate_monitoring_report(
            reference=pd.DataFrame({"Amount": [1.0], "fraud_probability": [0.1]}),
            current=pd.DataFrame({"fraud_probability": [0.2]}),
            feature_columns=("Amount",),
            score_column="fraud_probability",
        )


def test_monitoring_report_flags_prediction_and_feature_drift_alerts() -> None:
    from fraudshield.monitoring import AlertThresholds, generate_monitoring_report

    reference = pd.DataFrame(
        {
            "Amount": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "V1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "fraud_probability": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    current = pd.DataFrame(
        {
            "Amount": [10.5, 12.5, 14.5, 16.5, 18.5, 20.5],
            "V1": [0.0, 0.1, 0.2, 1.5, 1.7, 1.9],
            "fraud_probability": [0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
        }
    )

    report = generate_monitoring_report(
        reference=reference,
        current=current,
        feature_columns=("Amount", "V1"),
        score_column="fraud_probability",
        thresholds=AlertThresholds(warning=0.05, critical=0.20),
        bins=3,
    )

    assert report["schema_version"] == 1
    assert report["thresholds"] == {"warning": 0.05, "critical": 0.2}
    assert report["windows"] == {"reference_rows": 6, "current_rows": 6}
    assert report["status"] == "critical"
    metrics_by_name = {metric["name"]: metric for metric in report["metrics"]}
    assert metrics_by_name["prediction:fraud_probability"]["kind"] == "prediction_score"
    assert metrics_by_name["prediction:fraud_probability"]["status"] == "critical"
    assert metrics_by_name["feature:Amount"]["status"] == "ok"
    assert metrics_by_name["feature:V1"]["status"] in {"warning", "critical"}
    assert any(alert["name"] == "prediction:fraud_probability" for alert in report["alerts"])


def test_monitor_drift_cli_writes_versioned_report(tmp_path) -> None:
    from fraudshield.cli import app

    reference = pd.DataFrame(
        {
            "Amount": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "fraud_probability": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    current = pd.DataFrame(
        {
            "Amount": [10.5, 12.5, 14.5, 16.5, 18.5, 20.5],
            "fraud_probability": [0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
        }
    )
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    report_path = tmp_path / "drift_report.json"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "monitor-drift",
            str(reference_path),
            str(current_path),
            "--feature-column",
            "Amount",
            "--score-column",
            "fraud_probability",
            "--warning-threshold",
            "0.05",
            "--critical-threshold",
            "0.2",
            "--bins",
            "3",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Monitoring report:" in result.stdout
    assert "Overall status: critical" in result.stdout
    payload = pd.read_json(report_path, typ="series")
    assert payload["schema_version"] == 1
    assert payload["status"] == "critical"
    assert {metric["name"] for metric in payload["metrics"]} == {
        "feature:Amount",
        "prediction:fraud_probability",
    }
