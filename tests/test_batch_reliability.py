from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from typer.testing import CliRunner

from fraudshield.data import FEATURE_COLUMNS


class DeterministicProbabilityModel:
    """Small persisted test model with observable, order-sensitive predictions."""

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        assert tuple(frame.columns) == FEATURE_COLUMNS
        fraud = np.clip(frame["Amount"].to_numpy(dtype=float) / 1_000.0, 0.0, 1.0)
        return np.column_stack((1.0 - fraud, fraud))


class RecordingProbabilityModel:
    def __init__(self) -> None:
        self.calls = 0
        self.columns: tuple[str, ...] | None = None

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        self.calls += 1
        self.columns = tuple(frame.columns)
        fraud = np.arange(1, len(frame) + 1, dtype=float) / 10.0
        return np.column_stack((1.0 - fraud, fraud))


def _transaction(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {column: 0.0 for column in FEATURE_COLUMNS}
    values["Amount"] = 25.0
    values.update(overrides)
    return values


def _loaded_bundle(model: Any) -> Any:
    from fraudshield.serving import LoadedModelBundle, ModelBundleMetadata

    return LoadedModelBundle(
        model=model,
        metadata=ModelBundleMetadata(
            model_name="batch-fixture",
            model_version="2026.09-batch-v1",
            decision_threshold=0.5,
            decision_policy="review_when_probability_at_or_above_threshold",
            training_data_sha256="0" * 64,
        ),
        manifest={},
        bundle_dir=Path("."),
    )


def _write_bundle(tmp_path: Path) -> Path:
    from fraudshield.serving import ModelBundleMetadata, write_model_bundle

    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        model=DeterministicProbabilityModel(),
        bundle_dir=bundle_dir,
        metadata=ModelBundleMetadata(
            model_name="batch-fixture",
            model_version="2026.09-batch-v1",
            decision_threshold=0.5,
            decision_policy="review_when_probability_at_or_above_threshold",
            training_data_sha256="0" * 64,
        ),
    )
    return bundle_dir


def _write_csv(path: Path, rows: list[Mapping[str, Any]], *, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_batch_validation_is_bounded_aggregated_and_fail_closed() -> None:
    from fraudshield.serving import BATCH_MAX_ROWS, BatchScoringError

    model = RecordingProbabilityModel()
    bundle = _loaded_bundle(model)

    malformed = _transaction()
    malformed.pop("V1")
    invalid = _transaction(V2=float("nan"))
    invalid["unexpected"] = 4.0

    try:
        bundle.score_batch(
            [malformed, invalid],
            transaction_ids=["duplicate", "duplicate"],
        )
    except BatchScoringError as exc:
        message = str(exc)
        assert exc.schema_version == 1
        assert exc.issue_count == 4
        assert "batch validation failed with 4 issues" in message
        assert "duplicate transaction_id 'duplicate' at rows 0, 1" in message
        assert "row 0 missing required model features: V1" in message
        assert "row 1 contains unsupported model features: unexpected" in message
        assert "row 1 feature V2 must be finite" in message
    else:
        raise AssertionError("malformed batch was accepted")

    assert model.calls == 0

    try:
        bundle.score_batch([_transaction()] * (BATCH_MAX_ROWS + 1))
    except BatchScoringError as exc:
        assert str(exc) == "batch must contain at most 1000 transactions (received 1001)"
    else:
        raise AssertionError("oversized batch was accepted")
    assert model.calls == 0


def test_batch_predictions_preserve_input_order_and_exact_model_schema() -> None:
    model = RecordingProbabilityModel()
    bundle = _loaded_bundle(model)

    predictions = bundle.score_batch(
        [_transaction(Amount=900.0), _transaction(Amount=10.0), _transaction(Amount=500.0)],
        transaction_ids=["third", "first", "second"],
    )

    assert model.calls == 1
    assert model.columns == FEATURE_COLUMNS
    assert [prediction.transaction_id for prediction in predictions] == ["third", "first", "second"]
    assert [prediction.fraud_probability for prediction in predictions] == [0.1, 0.2, 0.3]


def test_model_bundle_rejects_corrupted_artifact_before_deserialization(tmp_path: Path) -> None:
    from fraudshield.serving import load_model_bundle

    bundle_dir = _write_bundle(tmp_path)
    (bundle_dir / "model.joblib").write_bytes(b"corrupted artifact")

    try:
        load_model_bundle(bundle_dir)
    except ValueError as exc:
        assert str(exc) == "model artifact SHA-256 does not match the bundle manifest"
    else:
        raise AssertionError("corrupted model artifact was loaded")


def test_cli_batch_contract_rejects_schema_duplicates_and_partial_output(tmp_path: Path) -> None:
    from fraudshield.cli import app

    bundle_dir = _write_bundle(tmp_path)
    output_path = tmp_path / "predictions.json"
    wrong_order_path = tmp_path / "wrong-order.csv"
    wrong_order = ["Amount", *[column for column in FEATURE_COLUMNS if column != "Amount"]]
    _write_csv(wrong_order_path, [_transaction()], fields=wrong_order)

    wrong_order_result = CliRunner().invoke(
        app,
        [
            "score-batch",
            str(wrong_order_path),
            "--model-bundle",
            str(bundle_dir),
            "--output-json",
            str(output_path),
        ],
    )
    assert wrong_order_result.exit_code == 2
    assert (
        "CSV header must exactly match the ordered scoring feature contract"
        in wrong_order_result.output
    )
    assert not output_path.exists()

    duplicate_path = tmp_path / "duplicate.csv"
    row = _transaction(Amount=300.0)
    _write_csv(duplicate_path, [row, row], fields=list(FEATURE_COLUMNS))
    duplicate_result = CliRunner().invoke(
        app,
        [
            "score-batch",
            str(duplicate_path),
            "--model-bundle",
            str(bundle_dir),
            "--output-json",
            str(output_path),
        ],
    )
    assert duplicate_result.exit_code == 2
    assert "duplicate input rows: 0, 1" in duplicate_result.output
    assert not output_path.exists()


def test_cli_and_api_batch_smokes_are_versioned_ordered_and_fail_closed(tmp_path: Path) -> None:
    from fraudshield.api import create_app
    from fraudshield.cli import app

    bundle_dir = _write_bundle(tmp_path)
    input_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "predictions.json"
    _write_csv(
        input_path,
        [_transaction(Amount=800.0), _transaction(Amount=100.0)],
        fields=list(FEATURE_COLUMNS),
    )

    cli_result = CliRunner().invoke(
        app,
        [
            "score-batch",
            str(input_path),
            "--model-bundle",
            str(bundle_dir),
            "--output-json",
            str(output_path),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    cli_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert cli_payload["schema_version"] == 1
    assert cli_payload["batch"] == {"input_rows": 2, "scored_rows": 2, "failed_rows": 0}
    assert [row["fraud_probability"] for row in cli_payload["predictions"]] == [0.8, 0.1]

    client = TestClient(create_app(bundle_dir))
    api_response = client.post(
        "/score-batch",
        json={
            "transactions": [
                {"transaction_id": "high", "features": _transaction(Amount=800.0)},
                {"transaction_id": "low", "features": _transaction(Amount=100.0)},
            ]
        },
    )
    assert api_response.status_code == 200, api_response.text
    api_payload = api_response.json()
    assert api_payload["schema_version"] == 1
    assert api_payload["batch"] == {"input_rows": 2, "scored_rows": 2, "failed_rows": 0}
    assert [row["transaction_id"] for row in api_payload["predictions"]] == ["high", "low"]
    assert [row["fraud_probability"] for row in api_payload["predictions"]] == [0.8, 0.1]

    rejected = client.post(
        "/score-batch",
        json={
            "transactions": [
                {"transaction_id": "same", "features": _transaction()},
                {"transaction_id": "same", "features": _transaction(Amount=900.0)},
            ]
        },
    )
    assert rejected.status_code == 422
    assert "duplicate transaction_id 'same' at rows 0, 1" in rejected.text
    assert "predictions" not in rejected.text
