"""Command-line interface for FraudShield."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Annotated, Protocol, cast

import joblib
import numpy as np
import pandas as pd
import typer
from numpy.typing import NDArray

from fraudshield.data import (
    DEFAULT_ARCHIVE_SHA256,
    DEFAULT_CSV_SHA256,
    DEFAULT_DATASET_SOURCE,
    FEATURE_COLUMNS,
    KAGGLE_DOWNLOAD_URL,
    TARGET_COLUMN,
    DatasetSource,
    prepare_kaggle_dataset,
    verify_dataset_file,
)
from fraudshield.decision import DecisionCosts, DecisionResult, optimize_decisions
from fraudshield.explainability import generate_explainability_report, render_model_card
from fraudshield.modeling import (
    chronological_train_holdout,
    compare_model_candidates,
    train_baseline,
)
from fraudshield.monitoring import (
    DEFAULT_MONITORING_BINS,
    DEFAULT_PSI_CRITICAL_THRESHOLD,
    DEFAULT_PSI_WARNING_THRESHOLD,
    AlertThresholds,
    generate_monitoring_report,
    write_monitoring_report,
)
from fraudshield.serving import load_model_bundle

app = typer.Typer(no_args_is_help=True, help="FraudShield model-development commands.")


class ProbabilityModel(Protocol):
    """Minimal prediction protocol for persisted fraud-risk models."""

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        """Return per-class probabilities for the supplied feature frame."""


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
    test_fraction: Annotated[
        float,
        typer.Option(help="Newest chronological fraction reserved for holdout evaluation."),
    ] = 0.2,
) -> None:
    """Train and persist the chronological logistic-regression baseline."""
    frame = pd.read_csv(dataset)
    result = train_baseline(frame, test_fraction=test_fraction)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "baseline.joblib"
    metrics_path = output_dir / "baseline_metrics.json"
    split_payload = {
        "strategy": "chronological",
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
        "test_fraction": result.test_fraction,
        "train_max_time": result.train_max_time,
        "holdout_min_time": result.holdout_min_time,
    }
    joblib.dump({"model": result.model, "split": split_payload}, model_path)
    metrics_payload = {
        "metrics": result.metrics,
        "split": split_payload,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    typer.echo(f"Model: {model_path}")
    typer.echo(f"Metrics: {metrics_path}")
    typer.echo(f"Average precision: {result.metrics['average_precision']:.6f}")
    typer.echo(f"Recall: {result.metrics['recall']:.6f}")


@app.command("evaluate-decisions")
def evaluate_decisions(
    dataset: Path,
    model_path: Annotated[
        Path,
        typer.Option(help="Ignored trained model artifact to score the holdout."),
    ],
    report_path: Annotated[
        Path,
        typer.Option(help="Machine-readable JSON decision-economics evidence path."),
    ] = Path("reports/decision_evidence.json"),
    test_fraction: Annotated[
        float,
        typer.Option(help="Newest chronological fraction reserved for holdout evaluation."),
    ] = 0.2,
    false_negative_cost: Annotated[
        float,
        typer.Option(help="Cost of missing a fraudulent transaction."),
    ] = 500.0,
    false_positive_cost: Annotated[
        float,
        typer.Option(help="Cost of sending a genuine transaction to analyst review."),
    ] = 25.0,
    analyst_alert_budget: Annotated[
        int,
        typer.Option(help="Maximum number of holdout transactions analysts can review."),
    ] = 100,
) -> None:
    """Score chronological holdout probabilities and write cost-sensitive decision evidence."""
    if not 0.0 < test_fraction < 1.0:
        raise typer.BadParameter("test_fraction must be between 0 and 1")
    frame = pd.read_csv(dataset)
    split = chronological_train_holdout(frame, test_fraction=test_fraction)
    holdout = split.holdout
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
            module="joblib.numpy_pickle",
        )
        loaded_model = joblib.load(model_path)
    model = _extract_probability_model(loaded_model, expected_train_max_time=split.train_max_time)
    feature_frame = holdout[list(FEATURE_COLUMNS)]
    probabilities = model.predict_proba(feature_frame)[:, 1]
    labels = tuple(int(label) for label in holdout[TARGET_COLUMN])
    decision = optimize_decisions(
        probabilities=tuple(float(probability) for probability in probabilities),
        labels=labels,
        costs=DecisionCosts(false_negative=false_negative_cost, false_positive=false_positive_cost),
        analyst_alert_budget=analyst_alert_budget,
    )
    payload = {
        "schema_version": 1,
        "dataset": {"rows": len(frame), "features": len(FEATURE_COLUMNS)},
        "model_artifact": {"format": "joblib", "tracked": False},
        "split": split.to_report(),
        "costs": {
            "false_negative": float(false_negative_cost),
            "false_positive": float(false_positive_cost),
        },
        "holdout_probability_count": len(probabilities),
        "decision": _decision_report(decision),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    typer.echo(f"Decision report: {report_path}")
    typer.echo(f"Chosen threshold: {decision.chosen_threshold:.6f}")
    typer.echo(f"Expected loss: {decision.expected_loss:.2f}")
    typer.echo(f"Holdout recall: {decision.recall:.6f}")


@app.command("compare-models")
def compare_models(
    dataset: Path,
    report_path: Annotated[
        Path,
        typer.Option(help="Versioned JSON model-comparison evidence path."),
    ] = Path("reports/model_comparison.json"),
    holdout_fraction: Annotated[
        float,
        typer.Option(help="Newest chronological fraction reserved for final holdout."),
    ] = 0.2,
    calibration_fraction: Annotated[
        float,
        typer.Option(help="Newest pre-holdout training fraction reserved for calibration."),
    ] = 0.2,
    false_negative_cost: Annotated[
        float,
        typer.Option(help="Cost of missing a fraudulent transaction."),
    ] = 500.0,
    false_positive_cost: Annotated[
        float,
        typer.Option(help="Cost of sending a genuine transaction to analyst review."),
    ] = 25.0,
    analyst_alert_budget: Annotated[
        int,
        typer.Option(help="Maximum number of holdout transactions analysts can review."),
    ] = 1_000,
) -> None:
    """Compare calibrated logistic and gradient-boosting candidates on a final holdout."""
    frame = pd.read_csv(dataset)
    result = compare_model_candidates(
        frame,
        holdout_fraction=holdout_fraction,
        calibration_fraction=calibration_fraction,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        analyst_alert_budget=analyst_alert_budget,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_report(), indent=2), encoding="utf-8")

    selected = result.evaluation_by_name(result.selected_model)
    typer.echo(f"Model comparison: {report_path}")
    typer.echo(f"Model comparison report: {report_path}")
    typer.echo(f"Selected model: {result.selected_model}")
    typer.echo(f"Selected PR-AUC: {selected.metrics['average_precision']:.6f}")
    typer.echo(f"Selected expected loss: {selected.decision.expected_loss:.2f}")


@app.command("explain-model")
def explain_model(
    dataset: Path,
    report_path: Annotated[
        Path,
        typer.Option(help="Machine-readable JSON explainability and error-analysis report path."),
    ] = Path("reports/explainability_report.json"),
    model_card_path: Annotated[
        Path,
        typer.Option(help="Human-readable model card Markdown path."),
    ] = Path("docs/model_card.md"),
    explanation_sample_size: Annotated[
        int,
        typer.Option(help="Highest-risk holdout rows to explain with feature perturbations."),
    ] = 25,
    analyst_alert_budget: Annotated[
        int,
        typer.Option(help="Maximum number of holdout transactions analysts can review."),
    ] = 1_000,
    holdout_fraction: Annotated[
        float,
        typer.Option(help="Newest chronological fraction reserved for final holdout."),
    ] = 0.2,
    calibration_fraction: Annotated[
        float,
        typer.Option(help="Newest pre-holdout training fraction reserved for calibration."),
    ] = 0.2,
    false_negative_cost: Annotated[
        float,
        typer.Option(help="Cost of missing a fraudulent transaction."),
    ] = 500.0,
    false_positive_cost: Annotated[
        float,
        typer.Option(help="Cost of sending a genuine transaction to analyst review."),
    ] = 25.0,
) -> None:
    """Write selected-model explainability, slice errors, and a model card."""
    frame = pd.read_csv(dataset)
    report = generate_explainability_report(
        frame,
        explanation_sample_size=explanation_sample_size,
        analyst_alert_budget=analyst_alert_budget,
        holdout_fraction=holdout_fraction,
        calibration_fraction=calibration_fraction,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_card_path.write_text(render_model_card(report), encoding="utf-8")

    typer.echo(f"Explainability report: {report_path}")
    typer.echo(f"Model card: {model_card_path}")
    typer.echo(f"Selected model: {report['selected_model']}")


@app.command("score-batch")
def score_batch_command(
    transactions_csv: Path,
    model_bundle: Annotated[
        Path,
        typer.Option(help="Local directory containing model-bundle.json and model.joblib."),
    ],
    output_json: Annotated[
        Path,
        typer.Option(help="JSON output file for ordered fraud predictions."),
    ],
) -> None:
    """Score a CSV of transaction features with a validated local model bundle."""
    bundle = load_model_bundle(model_bundle)
    frame = pd.read_csv(transactions_csv)
    transactions = [
        {column: float(row[column]) for column in FEATURE_COLUMNS}
        for row in frame.to_dict(orient="records")
    ]
    try:
        predictions = bundle.score_batch(transactions)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        "schema_version": 1,
        "model_version": bundle.metadata.model_version,
        "predictions": [prediction.to_dict() for prediction in predictions],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    typer.echo(f"Scored transactions: {len(predictions)}")
    typer.echo(f"Predictions: {output_json}")


@app.command("monitor-drift")
def monitor_drift(
    reference_csv: Path,
    current_csv: Path,
    feature_column: Annotated[
        list[str] | None,
        typer.Option(
            "--feature-column",
            "--feature",
            help="Numeric feature column to monitor; repeat for multiple columns.",
        ),
    ] = None,
    score_column: Annotated[
        str,
        typer.Option(
            "--score-column", "--prediction-column", help="Prediction score column to monitor."
        ),
    ] = "fraud_probability",
    report_path: Annotated[
        Path,
        typer.Option(help="Versioned JSON monitoring report path."),
    ] = Path("reports/monitoring_report.json"),
    warning_threshold: Annotated[
        float, typer.Option(help="PSI value that raises a warning alert.")
    ] = DEFAULT_PSI_WARNING_THRESHOLD,
    critical_threshold: Annotated[
        float,
        typer.Option(
            "--critical-threshold",
            "--alert-threshold",
            help="PSI value that raises a critical alert.",
        ),
    ] = DEFAULT_PSI_CRITICAL_THRESHOLD,
    bins: Annotated[
        int,
        typer.Option("--bins", "--bin-count", help="Number of reference-defined PSI buckets."),
    ] = DEFAULT_MONITORING_BINS,
) -> None:
    """Compare reference and current windows for feature and prediction drift."""
    reference_frame = pd.read_csv(reference_csv)
    current_frame = pd.read_csv(current_csv)
    feature_columns = (
        tuple(feature_column)
        if feature_column
        else _default_monitoring_features(reference_frame, current_frame)
    )
    if not feature_columns:
        raise typer.BadParameter("at least one feature column is required")
    try:
        report = generate_monitoring_report(
            reference=reference_frame,
            current=current_frame,
            feature_columns=feature_columns,
            score_column=score_column,
            thresholds=AlertThresholds(warning=warning_threshold, critical=critical_threshold),
            bins=bins,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    write_monitoring_report(report, report_path)

    typer.echo(f"Monitoring report: {report_path}")
    typer.echo(f"Overall status: {report['status']}")
    alerts = cast(list[object], report["alerts"])
    typer.echo(f"Alerts: {len(alerts)}")


def _default_monitoring_features(
    reference_frame: pd.DataFrame, current_frame: pd.DataFrame
) -> tuple[str, ...]:
    return tuple(
        column
        for column in FEATURE_COLUMNS
        if column in reference_frame.columns and column in current_frame.columns
    )


def _decision_report(decision: DecisionResult) -> dict[str, object]:
    return {
        "chosen_threshold": decision.chosen_threshold,
        "expected_loss": decision.expected_loss,
        "precision": decision.precision,
        "recall": decision.recall,
        "f1": decision.f1,
        "alert_count": decision.alert_count,
        "thresholds_evaluated": len(decision.threshold_sweep),
        "analyst_budget": {
            "alert_count": decision.budget_alert_count,
            "precision": decision.budget_precision,
            "recall": decision.budget_recall,
        },
    }


def _extract_probability_model(
    loaded_model: object, *, expected_train_max_time: float
) -> ProbabilityModel:
    if isinstance(loaded_model, dict) and "model" in loaded_model:
        split_payload = loaded_model.get("split")
        if isinstance(split_payload, dict):
            trained_max_time = split_payload.get("train_max_time")
            if (
                isinstance(trained_max_time, int | float)
                and float(trained_max_time) != expected_train_max_time
            ):
                raise typer.BadParameter(
                    "model split metadata does not match the requested chronological holdout"
                )
        return cast(ProbabilityModel, loaded_model["model"])
    return cast(ProbabilityModel, loaded_model)
