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

## Explainability and error analysis

The `explain-model` command trains the same candidate set, selects the recorded winner, and writes two explainability artifacts:

- [`../reports/explainability_report.json`](../reports/explainability_report.json), a versioned machine-readable report with selected-model metadata, leakage controls, global feature importance, sampled transaction explanations, and amount/time-window error slices.
- [`model_card.md`](model_card.md), a human-readable model card summarizing the selected model, intended use, feature privacy context, explanation method, slice diagnostics, and limitations.

The current explanation implementation uses a deterministic model-agnostic perturbation fallback rather than a model-specific SHAP dependency. For each sampled holdout row, it computes the calibrated fraud probability, replaces one model input at a time with the median value from the base model-training window, and records the probability delta. Positive deltas mean the observed feature value increased the calibrated fraud score relative to the model-train median; negative deltas mean it lowered the score.

Leakage controls in the report explicitly list the model input columns and baseline columns. `Class` is excluded from model inputs and counterfactual baselines; labels are used only after scoring to summarize holdout errors. The model card also distinguishes raw `Time`/`Amount` from PCA-anonymized `V1`-`V28` features, which limits business-level interpretation because the original merchant, cardholder, geography, and device fields are unavailable.
