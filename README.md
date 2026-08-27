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

The low default-threshold precision is expected and motivates the planned cost-aware threshold and review-capacity optimization. Machine-readable evidence is stored in [`reports/baseline_metrics.json`](reports/baseline_metrics.json).

## Target architecture

The baseline currently implements schema validation, chronological evaluation, and imbalance-aware logistic regression. The remaining components in this diagram are planned capabilities tracked in the roadmap.

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
        ├── cost-sensitive thresholding
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

### 2. Download the source dataset

```bash
curl -L "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud" -o data/raw/creditcardfraud.zip
unzip data/raw/creditcardfraud.zip -d data/raw
```

The dataset is intentionally excluded from Git. Verify its source and checksums in [`docs/dataset.md`](docs/dataset.md).

### 3. Validate the dataset

```bash
uv run fraudshield verify-data data/raw/creditcard.csv
```

Expected class balance:

```text
Transactions: 284,807
Fraud cases: 492
Fraud rate: 0.1727%
```

### 4. Train the baseline

```bash
uv run fraudshield train-baseline data/raw/creditcard.csv --output-dir artifacts
```

Model artifacts are generated locally and excluded from version control.

### 5. Run quality gates

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
