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
- Implemented: slice-aware model diagnostics for amount bands and chronological holdout windows
- Next: add drift monitoring for score distributions and feature-quality changes

## Explainability and reliability
- Implemented: deterministic global and transaction-level explanations with model-agnostic feature perturbations
- Implemented: versioned model card with anonymized PCA feature limitations and holdout error-analysis evidence
- Data-drift and prediction-drift monitoring
- Batch reliability checks for scoring outputs

## Serving and operations
- Typed batch-scoring interface
- FastAPI inference service with request validation
- Interactive analyst dashboard
- Containerized runtime, automated tests, linting, typing, and CI
