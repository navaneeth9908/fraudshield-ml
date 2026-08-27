# Dataset Card

## Source

- **Dataset:** Credit Card Fraud Detection
- **Publisher:** Machine Learning Group, Université Libre de Bruxelles (ULB)
- **Kaggle reference:** `mlg-ulb/creditcardfraud`
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Kaggle license label:** Database: Open Database, Contents: Database Contents
- **Last updated on Kaggle:** 2018-03-23

Review the current Kaggle dataset page and license terms before redistributing any source data. FraudShield does not commit the dataset to this repository.

## Observed snapshot

| Property | Value |
|---|---:|
| Rows | 284,807 |
| Fraud rows | 492 |
| Genuine rows | 284,315 |
| Fraud prevalence | 0.1727% |
| Input features | `Time`, `V1`–`V28`, `Amount` |
| Target | `Class` (1 = fraud, 0 = genuine) |

The `V1`–`V28` columns are anonymized principal components. `Time` is elapsed seconds from the first transaction, and `Amount` is the transaction amount.

## Integrity checksums

Checksums from the downloaded Kaggle snapshot used for baseline verification:

```text
creditcardfraud.zip  SHA-256 a0360ce715992212e9ac72d8ccdca97f4be87dc1fdf2bed011358f7ab409a28a
creditcard.csv       SHA-256 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

## Modeling constraints

- Severe class imbalance makes accuracy unsuitable as a primary metric.
- Anonymized components limit domain-level feature interpretation.
- The short historical window limits seasonality and long-term drift analysis.
- Random splitting can leak temporal information; FraudShield uses time-aware evaluation.
- Results from this dataset should not be interpreted as deployment readiness.
