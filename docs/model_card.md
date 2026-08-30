# FraudShield model card

## Model summary

- Selected model: `hist_gradient_boosting`
- Selection rule: PR-AUC first, then expected loss, recall, precision, and stable model-name tie break
- Chronological split: 182276 fit rows, 45569 calibration rows, 56962 final holdout rows
- Operating threshold: 0.008871 from minimum expected-loss analysis

## Intended use

FraudShield is an educational portfolio fraud-risk scoring system for offline analysis of the Kaggle credit-card fraud dataset. It demonstrates leakage-safe model comparison, calibrated probabilities, cost-sensitive thresholding, model-agnostic explanations, and holdout slice diagnostics. It is not a live payment authorization system.

## Features and privacy context

The model uses `Time`, `Amount`, and anonymized PCA components `V1` through `V28`. The `Class` target is excluded from model inputs and appears only in holdout evaluation/error-analysis aggregates. Because the PCA components are anonymized, explanations can rank component influence but cannot map those components back to customer, merchant, geography, or device concepts.

## Explanation method

Explanations use `model_agnostic_feature_perturbation` on a deterministic holdout sample of 25 rows from 56962 candidates. Each sampled row is rescored after replacing one feature with its model-train median. Positive deltas mean the observed value increased calibrated fraud probability relative to that model-train baseline; negative deltas mean it lowered risk.

| Rank | Feature | Mean absolute probability delta | Mean signed probability delta |
|---:|---|---:|---:|
| 1 | `V14` | 0.002860 | 0.002858 |
| 2 | `V17` | 0.001013 | 0.001011 |
| 3 | `V10` | 0.000377 | 0.000376 |
| 4 | `V12` | 0.000065 | 0.000065 |
| 5 | `Time` | 0.000004 | -0.000004 |
| 6 | `V4` | 0.000003 | -0.000000 |
| 7 | `V26` | 0.000002 | 0.000002 |
| 8 | `V20` | 0.000001 | 0.000001 |
| 9 | `V5` | 0.000001 | -0.000000 |
| 10 | `V8` | 0.000001 | -0.000000 |

## Slice-based error analysis

Amount bands:

| Slice | Rows | Fraud cases | Alerts | False positives | False negatives | Precision | Recall | Fraud rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `amount:[0, 25)` | 32053 | 49 | 73 | 33 | 9 | 0.5479 | 0.8163 | 0.0015 |
| `amount:[25, 100)` | 14814 | 11 | 19 | 9 | 1 | 0.5263 | 0.9091 | 0.0007 |
| `amount:[100, 500)` | 8480 | 11 | 16 | 7 | 2 | 0.5625 | 0.8182 | 0.0013 |
| `amount:[500, inf)` | 1615 | 4 | 10 | 7 | 1 | 0.3000 | 0.7500 | 0.0025 |

Time windows:

| Slice | Rows | Fraud cases | Alerts | False positives | False negatives | Precision | Recall | Fraud rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `time:early_holdout` | 18987 | 32 | 48 | 20 | 4 | 0.5833 | 0.8750 | 0.0017 |
| `time:middle_holdout` | 18988 | 31 | 42 | 16 | 5 | 0.6190 | 0.8387 | 0.0016 |
| `time:late_holdout` | 18987 | 12 | 28 | 20 | 4 | 0.2857 | 0.6667 | 0.0006 |

## Limitations and responsible use

- The data represents two days of European cardholder transactions from September 2013 and is not current production traffic.
- The PCA-anonymized `V1`-`V28` features limit human interpretability and fairness analysis.
- Perturbation explanations are practical, deterministic model-agnostic diagnostics, not causal claims.
- Error-analysis slices use labeled holdout data and should remain evaluation evidence, not features for fitting or calibration.
- A production deployment would require current labels, drift monitoring, review workflows, security controls, governance, and fairness evaluation.
