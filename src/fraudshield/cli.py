"""Command-line interface for FraudShield."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
import typer

from fraudshield.data import TARGET_COLUMN, validate_transactions
from fraudshield.modeling import train_baseline

app = typer.Typer(no_args_is_help=True, help="FraudShield model-development commands.")


@app.callback()
def main() -> None:
    """Run FraudShield model-development workflows."""


@app.command("verify-data")
def verify_data(dataset: Path) -> None:
    """Validate a transaction CSV and print its class balance."""
    frame = pd.read_csv(dataset)
    validate_transactions(frame)
    fraud_cases = int(frame[TARGET_COLUMN].sum())
    fraud_rate = fraud_cases / len(frame) * 100.0
    typer.echo(f"Transactions: {len(frame):,}")
    typer.echo(f"Fraud cases: {fraud_cases:,}")
    typer.echo(f"Fraud rate: {fraud_rate:.4f}%")


@app.command("train-baseline")
def train_baseline_command(
    dataset: Path,
    output_dir: Annotated[Path, typer.Option(help="Model artifact directory.")] = Path("artifacts"),
) -> None:
    """Train and persist the chronological logistic-regression baseline."""
    frame = pd.read_csv(dataset)
    result = train_baseline(frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "baseline.joblib"
    metrics_path = output_dir / "baseline_metrics.json"
    joblib.dump(result.model, model_path)
    metrics_payload = {
        "metrics": result.metrics,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    typer.echo(f"Model: {model_path}")
    typer.echo(f"Metrics: {metrics_path}")
    typer.echo(f"Average precision: {result.metrics['average_precision']:.6f}")
    typer.echo(f"Recall: {result.metrics['recall']:.6f}")
