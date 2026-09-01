from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from typer.testing import CliRunner

from fraudshield.data import FEATURE_COLUMNS


def _transaction(**overrides: float) -> dict[str, float]:
    values = {column: 0.0 for column in FEATURE_COLUMNS}
    values["Amount"] = 25.0
    values.update(overrides)
    return values


def _fitted_fixture_model() -> LogisticRegression:
    rows = []
    labels = []
    for index, amount in enumerate((5.0, 15.0, 250.0, 400.0, 800.0, 1200.0)):
        rows.append(_transaction(Time=float(index), V1=float(index) / 10.0, Amount=amount))
        labels.append(1 if amount >= 400.0 else 0)
    model = LogisticRegression(random_state=42)
    model.fit(pd.DataFrame(rows, columns=list(FEATURE_COLUMNS)), labels)
    return model


def _write_bundle(tmp_path: Path) -> Path:
    from fraudshield.serving import ModelBundleMetadata, write_model_bundle

    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        model=_fitted_fixture_model(),
        bundle_dir=bundle_dir,
        metadata=ModelBundleMetadata(
            model_name="fixture-logistic",
            model_version="2026.08-fixture",
            decision_threshold=0.40,
            decision_policy="review_when_probability_at_or_above_threshold",
            training_data_sha256="0" * 64,
        ),
    )
    return bundle_dir


def test_model_bundle_round_trips_versioned_contract_without_committed_binary(
    tmp_path: Path,
) -> None:
    from fraudshield.serving import load_model_bundle

    bundle_dir = _write_bundle(tmp_path)

    manifest = json.loads((bundle_dir / "model-bundle.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model"]["name"] == "fixture-logistic"
    assert manifest["model"]["version"] == "2026.08-fixture"
    assert manifest["model"]["artifact_path"] == "model.joblib"
    assert len(manifest["model"]["artifact_sha256"]) == 64
    assert manifest["features"] == list(FEATURE_COLUMNS)
    assert manifest["decision"]["threshold"] == 0.40

    bundle = load_model_bundle(bundle_dir)
    prediction = bundle.score_one(_transaction(Amount=900.0))
    assert 0.0 <= prediction.fraud_probability <= 1.0
    assert prediction.model_version == "2026.08-fixture"
    assert prediction.decision in {"review", "approve"}
    assert prediction.reason.startswith("fraud_probability")


def test_model_bundle_loading_suppresses_joblib_numpy_deprecation_warning(
    tmp_path: Path,
) -> None:
    from fraudshield.serving import load_model_bundle

    bundle_dir = _write_bundle(tmp_path)
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always")
        load_model_bundle(bundle_dir)

    assert [warning for warning in warnings_seen if warning.category is DeprecationWarning] == []


def test_fastapi_serving_endpoints_score_single_batch_and_reject_malformed_rows(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from fraudshield.api import create_app

    client = TestClient(create_app(_write_bundle(tmp_path)))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    readiness = client.get("/ready")
    assert readiness.status_code == 200
    assert readiness.json()["model_loaded"] is True

    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["schema_version"] == 1
    assert info.json()["model_name"] == "fixture-logistic"
    assert info.json()["feature_count"] == len(FEATURE_COLUMNS)

    single = client.post(
        "/score",
        json={"transaction_id": "txn-high", "features": _transaction(Amount=900.0)},
    )
    assert single.status_code == 200, single.text
    payload = single.json()
    assert payload["transaction_id"] == "txn-high"
    assert 0.0 <= payload["fraud_probability"] <= 1.0
    assert payload["decision"] in {"review", "approve"}
    assert payload["threshold"] == 0.40
    assert "fixture-logistic" in payload["reason"]

    batch = client.post(
        "/score-batch",
        json={
            "transactions": [
                {"transaction_id": "txn-low", "features": _transaction(Amount=10.0)},
                {"transaction_id": "txn-high", "features": _transaction(Amount=900.0)},
            ]
        },
    )
    assert batch.status_code == 200, batch.text
    assert [item["transaction_id"] for item in batch.json()["predictions"]] == [
        "txn-low",
        "txn-high",
    ]

    malformed = client.post(
        "/score",
        json={"transaction_id": "bad", "features": {"Amount": 12.0}},
    )
    assert malformed.status_code == 422
    assert "missing required model features" in malformed.text
    assert "traceback" not in malformed.text.lower()


def test_offline_batch_scoring_cli_writes_json_predictions(tmp_path: Path) -> None:
    from fraudshield.cli import app

    input_csv = tmp_path / "transactions.csv"
    output_json = tmp_path / "scores.json"
    rows = [_transaction(Time=1.0, Amount=20.0), _transaction(Time=2.0, Amount=950.0)]
    with input_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(FEATURE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    result = CliRunner().invoke(
        app,
        [
            "score-batch",
            str(input_csv),
            "--model-bundle",
            str(_write_bundle(tmp_path)),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Scored transactions: 2" in result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["model_version"] == "2026.08-fixture"
    assert len(payload["predictions"]) == 2
    assert all(0.0 <= row["fraud_probability"] <= 1.0 for row in payload["predictions"])
    assert {row["decision"] for row in payload["predictions"]}.issubset({"review", "approve"})


def test_dockerfile_uses_secure_nonroot_api_runtime() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:" in dockerfile
    assert "uvicorn" in dockerfile
    assert "FRAUDSHIELD_MODEL_BUNDLE" in dockerfile
    assert "USER appuser" in dockerfile
    assert "COPY --chown=appuser:appuser" in dockerfile
