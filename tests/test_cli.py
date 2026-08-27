from __future__ import annotations

import numpy as np
import pandas as pd
from typer.testing import CliRunner


def test_verify_data_reports_transaction_and_fraud_counts(tmp_path) -> None:
    from fraudshield.cli import app

    rows = 40
    frame = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{f"V{index}": np.zeros(rows) for index in range(1, 29)},
            "Amount": np.full(rows, 10.0),
            "Class": [1, 1, *([0] * (rows - 2))],
        }
    )
    dataset_path = tmp_path / "transactions.csv"
    frame.to_csv(dataset_path, index=False)

    result = CliRunner().invoke(app, ["verify-data", str(dataset_path)])

    assert result.exit_code == 0
    assert "Transactions: 40" in result.stdout
    assert "Fraud cases: 2" in result.stdout
    assert "Fraud rate: 5.0000%" in result.stdout


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
