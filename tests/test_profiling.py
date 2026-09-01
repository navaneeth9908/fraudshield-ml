from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner


def _profile_frame() -> pd.DataFrame:
    rows = 5
    return pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0, 4.0],
            **{f"V{index}": np.arange(index, index + rows, dtype=float) for index in range(1, 29)},
            "Amount": [0.0, 10.0, 10.0, 50.0, 100.0],
            "Class": [0, 0, 1, 0, 1],
        }
    )


def test_profile_report_is_versioned_deterministic_and_records_quality_evidence(
    tmp_path: Path,
) -> None:
    from fraudshield.data import file_sha256
    from fraudshield.profiling import generate_data_quality_report

    dataset_path = tmp_path / "transactions.csv"
    frame = _profile_frame()
    frame.to_csv(dataset_path, index=False)
    checksum = file_sha256(dataset_path)

    first = generate_data_quality_report(
        frame,
        dataset_path=dataset_path,
        csv_sha256=checksum,
        expected_csv_sha256=checksum,
        dataset_slug="fixture/transactions",
    )
    second = generate_data_quality_report(
        frame,
        dataset_path=dataset_path,
        csv_sha256=checksum,
        expected_csv_sha256=checksum,
        dataset_slug="fixture/transactions",
    )

    assert first == second
    assert first["schema_version"] == 1
    assert first["report_type"] == "fraudshield_data_quality"
    assert first["dataset"] == {
        "source": "fixture/transactions",
        "file_name": "transactions.csv",
        "raw_data_tracked": False,
    }
    assert first["integrity"] == {
        "status": "pass",
        "algorithm": "sha256",
        "expected": checksum,
        "actual": checksum,
    }
    assert first["quality"] == {
        "status": "pass",
        "schema": {
            "status": "pass",
            "required_columns": 31,
            "observed_columns": 31,
            "unexpected_columns": [],
        },
        "missing_values": {"status": "pass", "total": 0, "columns": {}},
        "non_finite_values": {"status": "pass", "total": 0, "columns": {}},
    }
    assert first["rows"] == {"total": 5, "genuine": 3, "fraud": 2}
    assert first["class_balance"] == {
        "genuine_fraction": 0.6,
        "fraud_fraction": 0.4,
        "imbalance_ratio_genuine_to_fraud": 1.5,
    }
    assert first["feature_summaries"]["Amount"] == {
        "min": 0.0,
        "mean": 34.0,
        "median": 10.0,
        "p95": 90.0,
        "max": 100.0,
    }
    assert tuple(first["feature_summaries"]) == (
        "Time",
        *(f"V{index}" for index in range(1, 29)),
        "Amount",
    )


def test_profile_data_cli_verifies_checksum_and_writes_repeatable_json(tmp_path: Path) -> None:
    from fraudshield.cli import app
    from fraudshield.data import file_sha256

    dataset_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "data_quality_report.json"
    _profile_frame().to_csv(dataset_path, index=False)
    checksum = file_sha256(dataset_path)

    args = [
        "profile-data",
        str(dataset_path),
        "--report-path",
        str(report_path),
        "--csv-sha256",
        checksum,
    ]
    first = CliRunner().invoke(app, args)
    first_bytes = report_path.read_bytes()
    second = CliRunner().invoke(app, args)

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    assert report_path.read_bytes() == first_bytes
    assert "Data quality report:" in first.stdout
    assert "Quality status: pass" in first.stdout
    assert "Transactions: 5" in first.stdout
    assert "Fraud cases: 2" in first.stdout
    assert checksum in first.stdout
