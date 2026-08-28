# FraudShield ML

[![CI](https://github.com/navaneeth9908/fraudshield-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/navaneeth9908/fraudshield-ml/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FraudShield is a production-oriented credit-card fraud risk-scoring system built on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It treats fraud detection as a constrained decision problem: catch costly fraud while controlling false alerts and analyst workload.

## Why this project

Only **492 of 284,807 transactions (0.1727%)** are fraudulent. Accuracy is therefore misleading—a model that predicts every transaction as genuine is more than 99.8% accurate. FraudShield emphasizes precision-recall performance, recall at operational review capacity, calibrated probabilities, and cost-sensitive thresholds.

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

## Target architecture

The baseline currently implements checksum-verified dataset preparation, strict schema validation, chronological evaluation, imbalance-aware logistic regression, and cost-sensitive decision evidence. The remaining components in this diagram are planned capabilities tracked in the roadmap.

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
Batch scoring / FastAPI / analyst dashboard
        │
        ▼
Drift and performance monitoring
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

### 4. Train the baseline

```bash
uv run fraudshield train-baseline data/raw/creditcard.csv --output-dir artifacts
```

Model artifacts are generated locally and excluded from version control.

### 5. Evaluate decision economics

```bash
uv run fraudshield evaluate-decisions data/raw/creditcard.csv \
  --model-path artifacts/baseline.joblib \
  --report-path reports/decision_evidence.json \
  --false-negative-cost 500 \
  --false-positive-cost 25 \
  --analyst-alert-budget 1000
```

The command reuses the chronological holdout split, verifies split metadata on bundled model artifacts, scores holdout probabilities, and writes aggregate JSON decision evidence.

### 6. Run quality gates

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=fraudshield --cov-branch -q
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
- Explicit model limitations and monitoring expectations

## Responsible-use limitations

This repository is an educational portfolio system, not a production payment authorization service. The source features are PCA-anonymized, the data represents two days of European cardholder transactions from September 2013, and real deployment would require current labeled data, governance, security controls, human review, fairness analysis, and continuous monitoring.

See [ROADMAP.md](ROADMAP.md) for the technical direction.
