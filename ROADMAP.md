# Product Roadmap

## Data foundation
- Implemented: reproducible Kaggle dataset preparation with ZIP/CSV SHA-256 verification
- Implemented: strict transaction schema validation for empty data, missing/null targets, invalid targets, null/non-finite values, and nonnumeric model features
- Implemented: leakage-safe chronological train/holdout partition for the baseline
- Next: add versioned data-quality reports for repeatable profiling evidence

## Modeling and decision intelligence
- Reproducible logistic-regression baseline
- Imbalance-aware gradient-boosting models
- Probability calibration and cost-sensitive threshold optimization
- Model comparison with precision-recall, recall-at-review-capacity, and expected-loss metrics

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
