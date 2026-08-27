"""Command-line interface for FraudShield."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
import typer

from fraudshield.data import (
    DEFAULT_ARCHIVE_SHA256,
    DEFAULT_CSV_SHA256,
    DEFAULT_DATASET_SOURCE,
    KAGGLE_DOWNLOAD_URL,
    DatasetSource,
    prepare_kaggle_dataset,
    verify_dataset_file,
)
from fraudshield.modeling import train_baseline

app = typer.Typer(no_args_is_help=True, help="FraudShield model-development commands.")


@app.callback()
def main() -> None:
    """Run FraudShield model-development workflows."""


@app.command("prepare-data")
def prepare_data(
    raw_dir: Annotated[Path, typer.Option(help="Ignored raw-data directory.")] = Path("data/raw"),
    archive_path: Annotated[
        Path | None,
        typer.Option(help="Existing Kaggle creditcardfraud.zip to verify instead of downloading."),
    ] = None,
    source_url: Annotated[
        str, typer.Option(help="Kaggle dataset download URL.")
    ] = KAGGLE_DOWNLOAD_URL,
    archive_sha256: Annotated[
        str,
        typer.Option(help="Expected SHA-256 for creditcardfraud.zip."),
    ] = DEFAULT_ARCHIVE_SHA256,
    csv_sha256: Annotated[
        str,
        typer.Option(help="Expected SHA-256 for extracted creditcard.csv."),
    ] = DEFAULT_CSV_SHA256,
    force_download: Annotated[
        bool, typer.Option(help="Refresh the archive when downloading.")
    ] = False,
) -> None:
    """Download or reuse the Kaggle archive, verify checksums, and validate creditcard.csv."""
    source = DatasetSource(
        slug=DEFAULT_DATASET_SOURCE.slug,
        download_url=source_url,
        archive_name=DEFAULT_DATASET_SOURCE.archive_name,
        archive_sha256=archive_sha256,
        csv_name=DEFAULT_DATASET_SOURCE.csv_name,
        csv_sha256=csv_sha256,
    )
    result = prepare_kaggle_dataset(
        raw_dir=raw_dir,
        archive_path=archive_path,
        expected_archive_sha256=archive_sha256,
        expected_csv_sha256=csv_sha256,
        source=source,
        force_download=force_download,
    )
    typer.echo(f"Prepared: {result.csv_path}")
    typer.echo(f"Transactions: {result.rows:,}")
    typer.echo(f"Fraud cases: {result.fraud_cases:,}")
    typer.echo(f"Fraud rate: {result.fraud_rate:.4f}%")
    typer.echo(f"Archive SHA-256: {result.archive_sha256}")
    typer.echo(f"CSV SHA-256: {result.csv_sha256}")


@app.command("verify-data")
def verify_data(
    dataset: Path,
    csv_sha256: Annotated[
        str | None,
        typer.Option(help="Optional expected SHA-256 for the CSV before schema validation."),
    ] = None,
) -> None:
    """Validate a transaction CSV and print its class balance."""
    result = verify_dataset_file(dataset, expected_csv_sha256=csv_sha256)
    typer.echo(f"Transactions: {result.rows:,}")
    typer.echo(f"Fraud cases: {result.fraud_cases:,}")
    typer.echo(f"Fraud rate: {result.fraud_rate:.4f}%")
    typer.echo(f"CSV SHA-256: {result.csv_sha256}")


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
