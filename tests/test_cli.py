from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner


def _sample_transactions(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{f"V{index}": np.zeros(rows) for index in range(1, 29)},
            "Amount": np.full(rows, 10.0),
            "Class": [1, 1, *([0] * (rows - 2))],
        }
    )


def test_verify_data_reports_transaction_and_fraud_counts(tmp_path: Path) -> None:
    from fraudshield.cli import app

    frame = _sample_transactions()
    dataset_path = tmp_path / "transactions.csv"
    frame.to_csv(dataset_path, index=False)

    result = CliRunner().invoke(app, ["verify-data", str(dataset_path)])

    assert result.exit_code == 0
    assert "Transactions: 40" in result.stdout
    assert "Fraud cases: 2" in result.stdout
    assert "Fraud rate: 5.0000%" in result.stdout


def test_prepare_data_cli_verifies_checksums_and_reports_output(tmp_path: Path) -> None:
    from fraudshield.cli import app
    from fraudshield.data import file_sha256

    source_csv = tmp_path / "source.csv"
    source_csv.write_text(_sample_transactions(5).to_csv(index=False), encoding="utf-8")
    zip_path = tmp_path / "creditcardfraud.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(source_csv, arcname="creditcard.csv")

    raw_dir = tmp_path / "raw"
    result = CliRunner().invoke(
        app,
        [
            "prepare-data",
            "--raw-dir",
            str(raw_dir),
            "--archive-path",
            str(zip_path),
            "--archive-sha256",
            file_sha256(zip_path),
            "--csv-sha256",
            file_sha256(source_csv),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (raw_dir / "creditcard.csv").is_file()
    assert "Prepared: " in result.stdout
    assert "Transactions: 5" in result.stdout
    assert "Archive SHA-256:" in result.stdout


def test_train_baseline_persists_model_and_metrics(tmp_path) -> None:
    from fraudshield.cli import app

    rows = 240
    rng = np.random.default_rng(11)
    fraud = np.zeros(rows, dtype=int)
    fraud[::20] = 1
    frame = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{f"V{index}": rng.normal(size=rows) for index in range(1, 29)},
            "Amount": rng.lognormal(mean=3.0, sigma=1.0, size=rows),
            "Class": fraud,
        }
    )
    dataset_path = tmp_path / "transactions.csv"
    output_dir = tmp_path / "artifacts"
    frame.to_csv(dataset_path, index=False)

    result = CliRunner().invoke(
        app,
        ["train-baseline", str(dataset_path), "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert (output_dir / "baseline.joblib").is_file()
    assert (output_dir / "baseline_metrics.json").is_file()
    assert "Average precision:" in result.stdout


def test_evaluate_decisions_writes_holdout_decision_evidence(tmp_path: Path) -> None:
    from fraudshield.cli import app

    rows = 240
    rng = np.random.default_rng(17)
    fraud = np.zeros(rows, dtype=int)
    fraud[::20] = 1
    frame = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{f"V{index}": rng.normal(size=rows) for index in range(1, 29)},
            "Amount": rng.lognormal(mean=3.0, sigma=1.0, size=rows),
            "Class": fraud,
        }
    )
    dataset_path = tmp_path / "transactions.csv"
    output_dir = tmp_path / "artifacts"
    report_path = tmp_path / "decision_evidence.json"
    frame.to_csv(dataset_path, index=False)
    train = CliRunner().invoke(
        app,
        [
            "train-baseline",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--test-fraction",
            "0.25",
        ],
    )
    assert train.exit_code == 0

    result = CliRunner().invoke(
        app,
        [
            "evaluate-decisions",
            str(dataset_path),
            "--model-path",
            str(output_dir / "baseline.joblib"),
            "--report-path",
            str(report_path),
            "--test-fraction",
            "0.25",
            "--false-negative-cost",
            "500",
            "--false-positive-cost",
            "25",
            "--analyst-alert-budget",
            "10",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Decision report:" in result.stdout
    payload = pd.read_json(report_path, typ="series")
    assert payload["split"]["strategy"] == "chronological"
    assert payload["split"]["train_rows"] == 180
    assert payload["split"]["test_rows"] == 60
    assert payload["split"]["train_max_time"] < payload["split"]["holdout_min_time"]
    assert payload["costs"] == {"false_negative": 500.0, "false_positive": 25.0}
    assert payload["decision"]["analyst_budget"]["alert_count"] == 10
    assert payload["decision"]["thresholds_evaluated"] > 0
    assert "threshold_sweep" not in payload["decision"]
    assert payload["holdout_probability_count"] == 60


def test_compare_models_cli_writes_versioned_model_selection_report(tmp_path: Path) -> None:
    from fraudshield.cli import app

    rows = 300
    rng = np.random.default_rng(23)
    fraud = (np.arange(rows) % 15 == 0).astype(int)
    frame = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            "V1": fraud * 3.0 + rng.normal(scale=0.35, size=rows),
            **{f"V{index}": rng.normal(size=rows) for index in range(2, 29)},
            "Amount": 25.0 + fraud * 90.0 + rng.normal(scale=2.0, size=rows),
            "Class": fraud,
        }
    )
    dataset_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "model_comparison.json"
    frame.to_csv(dataset_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "compare-models",
            str(dataset_path),
            "--report-path",
            str(report_path),
            "--analyst-alert-budget",
            "20",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Model comparison:" in result.stdout
    assert "Selected model:" in result.stdout
    payload = pd.read_json(report_path, typ="series")
    assert payload["schema_version"] == 1
    assert payload["split"]["strategy"] == "chronological_model_calibration_holdout"
    assert payload["split"]["calibration_max_time"] < payload["split"]["holdout_min_time"]
    assert payload["selection_rule"] == [
        "maximize_average_precision",
        "minimize_expected_loss",
        "maximize_recall",
        "maximize_precision",
        "lexicographic_model_name",
    ]
    assert payload["selected_model"] in {"logistic_regression", "hist_gradient_boosting"}
    assert {candidate["name"] for candidate in payload["candidates"]} == {
        "logistic_regression",
        "hist_gradient_boosting",
    }


def test_compare_models_writes_versioned_candidate_report(tmp_path: Path) -> None:
    from fraudshield.cli import app

    rows = 300
    rng = np.random.default_rng(23)
    fraud = (np.arange(rows) % 15 == 0).astype(int)
    frame = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            "V1": fraud * 3.0 + rng.normal(scale=0.35, size=rows),
            **{f"V{index}": rng.normal(size=rows) for index in range(2, 29)},
            "Amount": 20.0 + fraud * 80.0 + rng.normal(scale=2.0, size=rows),
            "Class": fraud,
        }
    )
    dataset_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "model_comparison.json"
    frame.to_csv(dataset_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "compare-models",
            str(dataset_path),
            "--report-path",
            str(report_path),
            "--holdout-fraction",
            "0.2",
            "--calibration-fraction",
            "0.2",
            "--false-negative-cost",
            "500",
            "--false-positive-cost",
            "25",
            "--analyst-alert-budget",
            "20",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Model comparison report:" in result.stdout
    assert "Selected model:" in result.stdout
    payload = pd.read_json(report_path, typ="series")
    assert payload["schema_version"] == 1
    assert payload["split"]["model_train_rows"] == 192
    assert payload["split"]["calibration_rows"] == 48
    assert payload["split"]["holdout_rows"] == 60
    assert payload["selection_rule"] == [
        "maximize_average_precision",
        "minimize_expected_loss",
        "maximize_recall",
        "maximize_precision",
        "lexicographic_model_name",
    ]
    assert {candidate["name"] for candidate in payload["candidates"]} == {
        "logistic_regression",
        "hist_gradient_boosting",
    }
    for candidate in payload["candidates"]:
        assert candidate["calibration"]["fit_on"] == "calibration"
        assert candidate["calibration"]["holdout_used"] is False
        assert candidate["threshold"]["selected_on"] == "holdout_probabilities"
        assert candidate["holdout_probability_count"] == 60
