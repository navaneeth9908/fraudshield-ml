# Product Roadmap

Status: **18 of 18 planned capabilities implemented**. Every item below maps to committed code, tests, or versioned aggregate evidence; raw Kaggle data and trained binaries remain local and ignored.

## Data foundation — 4/4
- Implemented: reproducible Kaggle dataset preparation with ZIP/CSV SHA-256 verification (`fraudshield prepare-data`, `docs/dataset.md`)
- Implemented: strict transaction schema validation for empty data, missing/null targets, invalid targets, null/non-finite values, and nonnumeric model features (`fraudshield verify-data`, `tests/test_data.py`)
- Implemented: leakage-safe chronological train/holdout partition for the baseline (`fraudshield train-baseline`, `tests/test_modeling.py`)
- Implemented: versioned, checksum-bound data-quality reports with deterministic class balance, schema, missing/non-finite, and feature-summary evidence (`reports/data_quality_report.json`, `tests/test_profiling.py`)

## Modeling and decision intelligence — 5/5
- Implemented: reproducible logistic-regression baseline (`reports/baseline_metrics.json`)
- Implemented: typed cost-sensitive decision layer with deterministic threshold sweep, expected-loss minimization, threshold precision/recall/F1, and analyst-budget precision/recall evidence (`reports/decision_evidence.json`, `tests/test_decisions.py`)
- Implemented: leakage-safe calibrated comparison of class-weighted logistic regression and histogram gradient boosting with deterministic selection evidence (`reports/model_comparison.json`, `tests/test_modeling.py`)
- Implemented: slice-aware model diagnostics for amount bands and chronological holdout windows (`reports/explainability_report.json`)
- Implemented: PSI drift monitoring for score distributions and feature-quality changes with configurable warning/critical thresholds (`reports/monitoring_report.json`, `tests/test_monitoring.py`)

## Explainability and reliability — 4/4
- Implemented: deterministic global and transaction-level explanations with model-agnostic feature perturbations (`reports/explainability_report.json`, `tests/test_explainability.py`)
- Implemented: versioned model card with anonymized PCA feature limitations and holdout error-analysis evidence (`docs/model_card.md`)
- Implemented: reference/current monitoring reports with stable bins, zero-count smoothing, and explicit alert evidence (`fraudshield monitor-drift`)
- Implemented: versioned, fail-closed batch scoring with bounded size, exact schema, duplicate/non-finite row checks, deterministic ordering, artifact integrity, and explicit row accounting (`tests/test_batch_reliability.py`)

## Serving and operations — 5/5
- Implemented: versioned model-bundle contract with SHA-256 artifact verification and fixed feature ordering (`fraudshield build-model-bundle`, `tests/test_serving.py`)
- Implemented: typed FastAPI health/readiness/model-info plus single and batch fraud scoring endpoints (`fraudshield.api`, `tests/test_serving.py`)
- Implemented: offline batch-scoring CLI with calibrated probability, decision, and reason metadata (`fraudshield score-batch`)
- Implemented: secure-default Docker API runtime with non-root execution (`Dockerfile`, `.github/workflows/ci.yml`)
- Implemented: dependency-light interactive analyst dashboard with single/batch demo flows, calibrated risk and decision context, model readiness, bounded errors, and responsible-use guardrails (`GET /dashboard`, `tests/test_dashboard.py`)

## Final acceptance contract

The canonical acceptance path is the CI-aligned local gate: `ruff format --check`, Ruff lint, strict mypy, full pytest with branch coverage, package build, dashboard/API/batch/data smokes, Docker build/run, secret and unwanted-artifact review, and `git diff --check`. GitHub Actions repeats formatting, lint, typing, branch coverage, package build, dashboard entry-point smoke, and container build/runtime smoke on `main`.
