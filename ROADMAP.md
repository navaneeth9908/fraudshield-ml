# Product Roadmap

## Data foundation
- Reproducible Kaggle dataset acquisition with checksum verification
- Strict transaction schema validation and data-quality reporting
- Leakage-safe train, validation, and test partitions

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
