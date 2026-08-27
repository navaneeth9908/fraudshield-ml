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
