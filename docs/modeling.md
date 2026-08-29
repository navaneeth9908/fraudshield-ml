# Advanced modeling evidence

FraudShield evaluates fraud-risk models with a nested chronological workflow so that each stage only sees past data:

1. `model_train` fits preprocessing and the base estimator.
2. `calibration` fits sigmoid probability calibration from base-model scores.
3. `holdout` remains untouched until final probability scoring, threshold sweep, business metrics, and model selection evidence.

The comparison command currently evaluates two deterministic, imbalance-aware candidates:

| Candidate | Key configuration |
|---|---|
| `logistic_regression` | `class_weight="balanced"`, `max_iter=1000`, `random_state=42` |
| `hist_gradient_boosting` | `class_weight="balanced"`, `early_stopping=False`, `learning_rate=0.08`, `max_iter=80`, `max_leaf_nodes=15`, `l2_regularization=0.01`, `random_state=42` |

Probability calibration uses the pre-holdout calibration slice only. Threshold evidence is computed from final holdout probabilities with the configured false-negative cost, false-positive review cost, and analyst alert budget.

The versioned report at [`../reports/model_comparison.json`](../reports/model_comparison.json) records:

- schema version and model-selection rule
- chronological fit/calibration/holdout boundaries
- deterministic model and threshold configuration
- PR-AUC, ROC-AUC, default-threshold precision/recall/F1
- expected-loss threshold metrics and analyst-budget recall/precision
- aggregate holdout probability summaries without storing model binaries or raw data

Current full-dataset run selected `hist_gradient_boosting` because it improved holdout PR-AUC and expected loss against the logistic baseline under the recorded selection rule.
