# Product Roadmap

## Data foundation
- Implemented: reproducible Kaggle dataset preparation with ZIP/CSV SHA-256 verification
- Implemented: strict transaction schema validation for empty data, missing/null targets, invalid targets, null/non-finite values, and nonnumeric model features
- Implemented: leakage-safe chronological train/holdout partition for the baseline
- Implemented: versioned, checksum-bound data-quality reports with deterministic class balance, schema, missing/non-finite, and feature-summary evidence

## Modeling and decision intelligence
- Implemented: reproducible logistic-regression baseline
- Implemented: typed cost-sensitive decision layer with deterministic threshold sweep, expected-loss minimization, threshold precision/recall/F1, and analyst-budget precision/recall evidence
- Implemented: leakage-safe calibrated comparison of class-weighted logistic regression and histogram gradient boosting with deterministic selection evidence
- Implemented: slice-aware model diagnostics for amount bands and chronological holdout windows
- Implemented: PSI drift monitoring for score distributions and feature-quality changes with configurable warning/critical thresholds

## Explainability and reliability
- Implemented: deterministic global and transaction-level explanations with model-agnostic feature perturbations
- Implemented: versioned model card with anonymized PCA feature limitations and holdout error-analysis evidence
- Implemented: reference/current monitoring reports with stable bins, zero-count smoothing, and explicit alert evidence
- Batch reliability checks for scoring outputs

## Serving and operations
- Implemented: versioned model-bundle contract with SHA-256 artifact verification and fixed feature ordering
- Implemented: typed FastAPI health/readiness/model-info plus single and batch fraud scoring endpoints
- Implemented: offline batch-scoring CLI with calibrated probability, decision, and reason metadata
- Implemented: secure-default Docker API runtime with non-root execution
- Interactive analyst dashboard
