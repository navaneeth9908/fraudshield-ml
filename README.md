# FraudShield ML

[![CI](https://github.com/navaneeth9908/fraudshield-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/navaneeth9908/fraudshield-ml/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FraudShield is a production-oriented credit-card fraud risk-scoring system built on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It treats fraud detection as a constrained decision problem: catch costly fraud while controlling false alerts and analyst workload.

## Why this project

Only **492 of 284,807 transactions (0.1727%)** are fraudulent. Accuracy is therefore misleading—a model that predicts every transaction as genuine is more than 99.8% accurate. FraudShield emphasizes precision-recall performance, recall at operational review capacity, calibrated probabilities, and cost-sensitive thresholds.

## Data-quality evidence

FraudShield profiles the exact checksum-verified Kaggle snapshot before modeling. The versioned report is deterministic—there is no wall-clock metadata—and records the source slug, expected and observed CSV SHA-256, schema/missing/non-finite checks, class balance, and min/mean/median/p95/max summaries for all 30 model features.

| Profiling evidence | Observed value |
|---|---:|
| Schema version | 1 |
| Rows | 284,807 |
| Required / observed columns | 31 / 31 |
| Missing / non-finite values | 0 / 0 |
| Genuine / fraud rows | 284,315 / 492 |
| Genuine-to-fraud ratio | 577.8760:1 |
| Amount median / p95 / maximum | 22.00 / 365.00 / 25,691.16 |

Machine-readable evidence is stored in [`reports/data_quality_report.json`](reports/data_quality_report.json). The report contains aggregate statistics and checksum provenance only; the 150 MB raw CSV and source archive remain ignored.

## Current baseline

The first reproducible model is a class-weighted logistic-regression pipeline evaluated on the newest 20% of transactions by `Time`, keeping future observations out of training.

| Holdout metric | Value |
|---|---:|
| Average precision (PR-AUC) | 0.7611 |
| ROC-AUC | 0.9863 |
| Recall at 0.50 threshold | 0.8933 |
| Precision at 0.50 threshold | 0.0707 |
| F1 at 0.50 threshold | 0.1311 |

The low default-threshold precision is expected, so FraudShield now evaluates holdout probabilities with a cost-sensitive decision layer. Machine-readable baseline evidence is stored in [`reports/baseline_metrics.json`](reports/baseline_metrics.json).

## Decision economics evidence

The decision layer sweeps deterministic probability thresholds, prices false negatives and false positives, chooses the minimum expected-loss threshold, and reports precision/recall/F1 at that operating point. It also reports recall and precision for a fixed analyst alert budget.

Current holdout decision evidence uses a missed-fraud cost of `500`, a false-alert review cost of `25`, and an analyst budget of `1,000` alerts:

| Decision metric | Value |
|---|---:|
| Chosen threshold | 0.9810 |
| Expected loss | 8,400.00 |
| Alerts at chosen threshold | 117 |
| Precision at chosen threshold | 0.5214 |
| Recall at chosen threshold | 0.8133 |
| F1 at chosen threshold | 0.6354 |
| Recall at 1,000-alert budget | 0.8933 |
| Precision at 1,000-alert budget | 0.0670 |

Machine-readable decision evidence is stored in [`reports/decision_evidence.json`](reports/decision_evidence.json). The report records split boundaries and aggregate probability evidence, but it does not track source data or model binaries.

## Advanced model comparison

FraudShield now compares two imbalance-aware candidates with a leakage-safe nested chronological split: base estimators and preprocessing fit on the oldest training window, sigmoid probability calibration fits on a later calibration window, and the newest holdout is reserved for final PR-AUC, threshold, and business-metric evidence.

The implemented candidates are a class-weighted logistic-regression baseline and a deterministic class-weighted histogram gradient-boosting model. Selection uses the versioned rule recorded in the report: maximize holdout average precision, then minimize expected loss, then maximize recall, then maximize precision, with a stable model-name tie break.

Current full-dataset comparison evidence uses the Kaggle CSV, a 20% final holdout, a 20% calibration slice from the pre-holdout rows, missed-fraud cost `500`, false-alert review cost `25`, and a `1,000` alert budget:

| Candidate | PR-AUC | ROC-AUC | Expected loss | Threshold | Precision | Recall | Budget recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.7368 | 0.9843 | 8,500.00 | 0.1403 | 0.3865 | 0.8400 | 0.8800 |
| Histogram gradient boosting | 0.7698 | 0.9836 | 7,900.00 | 0.0089 | 0.5254 | 0.8267 | 0.8800 |

The selected candidate is `hist_gradient_boosting`. Machine-readable comparison evidence is stored in [`reports/model_comparison.json`](reports/model_comparison.json), including deterministic model configuration, split boundaries, calibration provenance, threshold settings, PR-AUC/ROC-AUC, holdout probability summaries, and cost-sensitive decision metrics. The report does not track source data or model binaries.

## Explainability and error analysis

FraudShield now explains the selected calibrated model with a deterministic model-agnostic perturbation workflow. The command samples representative holdout rows by calibrated-risk quantiles, replaces one feature at a time with its model-train median, and records how each perturbation changes the fraud probability. This avoids a hard dependency on model-specific SHAP internals while producing reproducible global and transaction-level explanation evidence for either supported candidate model.

The current full-dataset report uses a 25-row holdout sample for runtime safety and selected `hist_gradient_boosting`:

| Global rank | Feature | Mean absolute probability delta | Mean signed probability delta |
|---:|---|---:|---:|
| 1 | `V14` | 0.002860 | 0.002858 |
| 2 | `V17` | 0.001013 | 0.001011 |
| 3 | `V10` | 0.000377 | 0.000376 |
| 4 | `V12` | 0.000065 | 0.000065 |
| 5 | `Time` | 0.000004 | -0.000004 |

The report also aggregates false positives, false negatives, precision, recall, alerts, and fraud rate by amount bands (`[0,25)`, `[25,100)`, `[100,500)`, `[500,∞)`) and early/middle/late holdout time windows. Machine-readable evidence is stored in [`reports/explainability_report.json`](reports/explainability_report.json), and the human-readable model card is stored in [`docs/model_card.md`](docs/model_card.md). Both documents explicitly distinguish raw `Time`/`Amount` from anonymized PCA features `V1`-`V28` and describe interpretability limitations.

## Production inference interface

FraudShield now includes a production-style serving layer around locally generated model bundles. A bundle stores `model-bundle.json` beside an ignored `model.joblib` artifact, pins `schema_version: 1`, records the exact ordered `Time`, `V1`-`V28`, and `Amount` feature contract, verifies the model artifact SHA-256 before loading, and keeps trained binaries out of Git.

The serving layer exposes:

- `GET /health` for process liveness
- `GET /ready` for loaded-model readiness
- `GET /model-info` for public bundle metadata
- `POST /score` for one transaction
- `POST /score-batch` for bounded batch scoring

Each prediction returns a calibrated fraud probability, the configured threshold, a `review` or `approve` decision, model version, and human-readable reason metadata. Malformed payloads, missing model features, unsupported extra features, and non-finite feature values fail closed with validation errors rather than tracebacks.

Batch scoring uses a versioned `schema_version: 1` contract and accepts at most 1,000 transactions. The service validates the complete batch before calling the model, aggregates row-level errors, rejects repeated feature rows and duplicate transaction IDs, forces the exact model feature order, and validates the complete probability matrix before returning any predictions. Successful CLI and API responses preserve input order and report `input_rows`, `scored_rows`, and `failed_rows: 0`; failures return no partial prediction payload. The CLI additionally requires the exact ordered CSV header and writes output through an atomic replacement only after every row succeeds.

The same contract is available offline through `fraudshield score-batch`, which reads a feature-only CSV and writes ordered JSON prediction evidence. Docker packaging runs the FastAPI app as a non-root user and expects `FRAUDSHIELD_MODEL_BUNDLE` to point at a mounted/generated bundle directory. See [`docs/architecture.md`](docs/architecture.md) for the serving-contract and runtime design.

## Interactive analyst dashboard

`GET /dashboard` serves a self-contained local analyst console from the existing FastAPI runtime, so the project adds no browser framework or external telemetry. The responsive interface includes synthetic single-transaction presets, editable exact-schema JSON, calibrated fraud probability and review-decision presentation, minimum expected-loss threshold context, a bounded batch-evidence table, model metadata/readiness, and explicit responsible-use guardrails. Invalid JSON, unavailable models, contract violations, and service failures are shown as inline errors without injecting API output as HTML.

The dashboard and API use the same validated model bundle and scoring methods. `fraudshield build-model-bundle` verifies the source CSV checksum, retrains the deterministic calibrated candidate comparison, and persists the selected model with its operating threshold and metadata. Its default destination is the ignored `artifacts/model-bundle` directory; custom `--bundle-dir` destinations must also remain untracked. Raw Kaggle rows and trained binaries remain untracked.

## Monitoring evidence

FraudShield now includes a lightweight monitoring foundation for reference/current window checks. The first production-oriented report uses population stability index (PSI) with stable bins learned from the reference window, smoothing for zero-count buckets, and configurable warning/critical thresholds. It monitors both model features and calibrated prediction-score distributions without requiring labels in the current window.

The checked-in monitoring evidence compares the calibration window with the final chronological holdout for all `Time`, `V1`-`V28`, `Amount`, and `fraud_probability` distributions:

| Monitoring metric | Value |
|---|---:|
| Metrics evaluated | 31 |
| Alert thresholds | warning `0.10`, critical `0.25` PSI |
| Current status | `critical` |
| Current alerts | 1 |
| Highest PSI signal | `feature:Time` at 22.1672 |

The critical `Time` drift is expected for this backtest because the reference and current windows are deliberately chronological and non-overlapping. The report is useful as an operational contract and alerting example, not as proof of live degradation. Machine-readable evidence is stored in [`reports/monitoring_report.json`](reports/monitoring_report.json).

## Implemented architecture

FraudShield implements checksum-verified dataset preparation, strict schema validation, chronological evaluation, imbalance-aware logistic regression, calibrated gradient-boosting comparison, cost-sensitive decision evidence, model-agnostic explanations, slice-based error analysis, versioned model bundles, offline batch scoring, FastAPI inference, a local analyst dashboard, a containerized API runtime, and PSI monitoring reports.

```text
Kaggle transactions
        │
        ▼
Schema and quality validation
        │
        ▼
Chronological train/holdout split
        │
        ▼
Imbalance-aware model pipeline
        │
        ├── probability calibration
        ├── cost-sensitive thresholding and analyst-budget evidence
        └── explainability
        │
        ▼
Versioned model bundle
        │
        ├── offline batch scoring CLI
        ├── FastAPI scoring service
        └── local analyst dashboard
        │
        ▼
PSI drift monitoring reports
```

## Quick start

### 1. Install

```bash
uv sync --group dev
```

### 2. Prepare the source dataset

```bash
uv run fraudshield prepare-data
```

This downloads or reuses `data/raw/creditcardfraud.zip`, verifies the known Kaggle ZIP and extracted CSV SHA-256 checksums, extracts `creditcard.csv`, and runs the transaction schema contract. The dataset is intentionally excluded from Git; source, license, and checksum details are in [`docs/dataset.md`](docs/dataset.md).

### 3. Validate the dataset

```bash
uv run fraudshield verify-data data/raw/creditcard.csv \
  --csv-sha256 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

The validator rejects empty files, missing or null `Class` targets, invalid target values, null or non-finite numeric values, and nonnumeric model features before training.

Expected class balance:

```text
Transactions: 284,807
Fraud cases: 492
Fraud rate: 0.1727%
CSV SHA-256: 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

### 4. Generate repeatable data-quality evidence

```bash
uv run fraudshield profile-data data/raw/creditcard.csv \
  --report-path reports/data_quality_report.json
```

The command verifies the known Kaggle CSV checksum before profiling and writes the versioned aggregate report described above. Repeating it against the same snapshot produces identical JSON.

### 5. Train the baseline

```bash
uv run fraudshield train-baseline data/raw/creditcard.csv --output-dir artifacts
```

Model artifacts are generated locally and excluded from version control.

### 6. Evaluate decision economics

```bash
uv run fraudshield evaluate-decisions data/raw/creditcard.csv \
  --model-path artifacts/baseline.joblib \
  --report-path reports/decision_evidence.json \
  --false-negative-cost 500 \
  --false-positive-cost 25 \
  --analyst-alert-budget 1000
```

The command reuses the chronological holdout split, verifies split metadata on bundled model artifacts, scores holdout probabilities, and writes aggregate JSON decision evidence.

### 7. Compare calibrated model candidates

```bash
uv run fraudshield compare-models data/raw/creditcard.csv \
  --report-path reports/model_comparison.json \
  --holdout-fraction 0.2 \
  --calibration-fraction 0.2 \
  --false-negative-cost 500 \
  --false-positive-cost 25 \
  --analyst-alert-budget 1000
```

The comparison command trains the logistic and gradient-boosting candidates with fixed seeds, calibrates probabilities on pre-holdout calibration rows only, scores the final chronological holdout, and writes a versioned JSON report with model-selection evidence.

### 8. Generate explainability evidence and the model card

```bash
uv run fraudshield explain-model data/raw/creditcard.csv \
  --report-path reports/explainability_report.json \
  --model-card-path docs/model_card.md \
  --explanation-sample-size 25 \
  --analyst-alert-budget 1000
```

The command retrains the deterministic calibrated candidates, selects the best holdout model under the recorded selection rule, generates global and transaction-level perturbation explanations for a representative holdout sample, and writes amount/time slice error analysis. It does not write source data or model binaries.

### 9. Score transactions with a local model bundle

Generated bundles contain an ignored `model.joblib` plus a versioned `model-bundle.json` contract. They can be mounted into the API container or used by the offline CLI without committing trained binaries.

```bash
uv run fraudshield score-batch data/processed/scoring_sample.csv \
  --model-bundle artifacts/model-bundle \
  --output-json reports/scoring_sample_predictions.json
```

The output JSON includes `schema_version`, `model_version`, explicit all-or-nothing batch row counts, and one ordered prediction per input row with calibrated fraud probability, decision threshold, `review`/`approve` decision, and reason metadata. Input must use the exact ordered `Time`, `V1`-`V28`, `Amount` header, contain no duplicate rows, and stay within the 1,000-row limit.

To serve the same bundle locally:

```bash
FRAUDSHIELD_MODEL_BUNDLE=artifacts/model-bundle \
  uv run uvicorn fraudshield.api:app --host 127.0.0.1 --port 8000
```

The API provides `/health`, `/ready`, `/model-info`, `/score`, and `/score-batch`. Docker builds the same runtime with a non-root application user:

```bash
docker build -t fraudshield-ml:local .
```

### 10. Build and open the analyst dashboard

Build a local bundle from the checksum-verified dataset. Both generated files remain under ignored `artifacts/`:

```bash
uv run fraudshield build-model-bundle data/raw/creditcard.csv \
  --bundle-dir artifacts/model-bundle \
  --model-version 0.1.0

FRAUDSHIELD_MODEL_BUNDLE=artifacts/model-bundle \
  uv run uvicorn fraudshield.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/dashboard>. The readiness panel stays usable when the bundle is absent or invalid and explains how to recover; scoring actions fail closed through the API. Dashboard presets are synthetic demonstrations, not known genuine/fraud labels.

### 11. Monitor reference/current drift

`monitor-drift` compares two scored windows. Each CSV needs the monitored feature columns plus a prediction score column:

```bash
uv run fraudshield monitor-drift reference_window.csv current_window.csv \
  --feature-column Time \
  --feature-column Amount \
  --score-column fraud_probability \
  --warning-threshold 0.10 \
  --critical-threshold 0.25 \
  --report-path reports/monitoring_report.json
```

The resulting JSON records reference/current row counts, per-bucket PSI components, alert statuses, thresholds, and limitations.

### 12. Run quality gates

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=fraudshield --cov-branch -q
uv build
```

## Repository layout

```text
fraudshield-ml/
├── src/fraudshield/       # Validated package and CLI
├── tests/                 # Unit and workflow tests
├── data/                  # Ignored raw/processed data locations
├── artifacts/             # Ignored trained model artifacts
├── reports/               # Versioned evaluation evidence
├── docs/                  # Dataset and design documentation
├── .github/workflows/     # CI quality gates
└── ROADMAP.md             # Planned technical capabilities
```

## Engineering principles

- Leakage-safe, time-aware evaluation
- Precision-recall and business-cost metrics over accuracy
- Test-first behavioral changes
- Reproducible environments and deterministic seeds
- Typed interfaces and automated quality gates
- Explicit model limitations and monitoring evidence

## Responsible-use limitations

This repository is an educational portfolio system, not a production payment authorization service. A dashboard `review` result is decision support for a human analyst and must not automatically block a payment. The source features are PCA-anonymized, the data represents two days of European cardholder transactions from September 2013, and real deployment would require current labeled data, governance, security controls, human review and appeals, fairness analysis, and continuous monitoring tied to live operating procedures.

See [ROADMAP.md](ROADMAP.md) for the technical direction.
