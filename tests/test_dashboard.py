from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_decision_view_explains_probability_against_operating_threshold() -> None:
    from fraudshield.dashboard import build_decision_view

    review = build_decision_view(probability=0.73, threshold=0.40)
    assert review.decision == "review"
    assert review.probability_percent == "73.00%"
    assert review.threshold_percent == "40.00%"
    assert review.margin_percent == "+33.00 pp"
    assert "human review" in review.summary

    approve = build_decision_view(probability=0.12, threshold=0.40)
    assert approve.decision == "approve"
    assert approve.margin_percent == "-28.00 pp"
    assert "below" in approve.summary

    with pytest.raises(ValueError, match="probability"):
        build_decision_view(probability=1.01, threshold=0.40)
    with pytest.raises(ValueError, match="threshold"):
        build_decision_view(probability=0.30, threshold=-0.01)


def test_dashboard_html_exposes_single_batch_evidence_and_responsible_use() -> None:
    from fraudshield.dashboard import render_dashboard

    html = render_dashboard()

    assert html.startswith("<!doctype html>")
    assert html.count("<h1") == 1
    assert 'id="model-readiness"' in html
    assert 'id="single-input"' in html
    assert 'id="score-single"' in html
    assert 'id="batch-input"' in html
    assert 'id="score-batch"' in html
    assert 'id="single-error"' in html
    assert 'id="batch-error"' in html
    assert 'aria-live="polite"' in html
    assert "/model-info" in html
    assert "/score" in html
    assert "/score-batch" in html
    assert "Calibrated fraud probability" in html
    assert "Minimum expected-loss threshold" in html
    assert "$500" in html
    assert "$25" in html
    assert "1,000 alerts" in html
    assert "No automatic payment blocking" in html
    assert "PCA-anonymized" in html
    assert "not a production payment authorization service" in html
    assert "innerHTML" not in html


def test_dashboard_route_is_a_dependency_light_application_entry_point() -> None:
    from fraudshield.api import create_app

    client = TestClient(create_app())

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.headers["content-type"].startswith("text/html")
    assert "FraudShield analyst console" in dashboard.text

    readiness = client.get("/ready")
    assert readiness.status_code == 503
    assert "model bundle is not loaded" in readiness.text


def test_ci_smoke_covers_dashboard_entry_point() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "TestClient" in workflow
    assert "'/dashboard'" in workflow or '"/dashboard"' in workflow


def test_build_model_bundle_cli_creates_dashboard_ready_calibrated_bundle(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd
    from typer.testing import CliRunner

    from fraudshield.cli import app
    from fraudshield.data import file_sha256
    from fraudshield.serving import load_model_bundle

    rows = 300
    rng = np.random.default_rng(47)
    fraud = (np.arange(rows) % 15 == 0).astype(int)
    frame = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            "V1": fraud * 3.0 + rng.normal(scale=0.35, size=rows),
            **{f"V{index}": rng.normal(size=rows) for index in range(2, 29)},
            "Amount": 25.0 + fraud * 100.0 + rng.normal(scale=2.0, size=rows),
            "Class": fraud,
        }
    )
    dataset = tmp_path / "transactions.csv"
    bundle_dir = tmp_path / "model-bundle"
    frame.to_csv(dataset, index=False)

    result = CliRunner().invoke(
        app,
        [
            "build-model-bundle",
            str(dataset),
            "--bundle-dir",
            str(bundle_dir),
            "--csv-sha256",
            file_sha256(dataset),
            "--model-version",
            "test-2026.09",
            "--analyst-alert-budget",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dashboard model bundle:" in result.output
    assert "Calibration: sigmoid" in result.output
    bundle = load_model_bundle(bundle_dir)
    assert bundle.metadata.model_version == "test-2026.09"
    assert bundle.metadata.calibration_method == "sigmoid"
    assert 0.0 <= bundle.metadata.decision_threshold <= 1.0
    assert bundle.model_info()["decision_policy"] == (
        "review_when_calibrated_probability_at_or_above_minimum_expected_loss_threshold"
    )
