# Product Roadmap

## Data foundation
- Implemented: reproducible Kaggle dataset preparation with ZIP/CSV SHA-256 verification
- Implemented: strict transaction schema validation for empty data, missing/null targets, invalid targets, null/non-finite values, and nonnumeric model features
- Implemented: leakage-safe chronological train/holdout partition for the baseline
- Next: add versioned data-quality reports for repeatable profiling evidence

## Modeling and decision intelligence
- Implemented: reproducible logistic-regression baseline
- Implemented: typed cost-sensitive decision layer with deterministic threshold sweep, expected-loss minimization, threshold precision/recall/F1, and analyst-budget precision/recall evidence
- Implemented: leakage-safe calibrated comparison of class-weighted logistic regression and histogram gradient boosting with deterministic selection evidence
- Next: add slice-aware model diagnostics for fraud-rate, amount, and time windows

## Explainability and reliability
- Global and transaction-level explanations
- Slice-based error analysis
- Data-drift and prediction-drift monitoring
- Versioned model cards and evaluation reports

## Serving and operations
- Typed batch-scoring interface
- FastAPI inference service with request validation
- Interactive analyst dashboard
- Containerized runtime, automated tests, linting, typing, and CI
