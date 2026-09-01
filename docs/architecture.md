# FraudShield serving architecture

FraudShield's serving layer turns the evaluated fraud model workflow into a typed, local-first inference runtime without committing trained binaries or secrets.

## Model-bundle contract

A serving bundle is a directory containing:

- `model-bundle.json` — versioned manifest with `schema_version: 1`.
- `model.joblib` — ignored local model artifact referenced by the manifest.

The manifest records the model name/version, expected ordered feature list (`Time`, `V1`-`V28`, `Amount`), calibration method, decision policy, operating threshold, training-data SHA-256, model artifact format, and artifact SHA-256. Loading verifies the manifest shape, the exact feature contract, that the artifact path stays inside the bundle directory, and that the artifact digest matches before deserialization.

## Prediction service

`fraudshield.serving.LoadedModelBundle` provides the typed scoring interface:

- validates every scoring row has all required features and no unsupported extras;
- rejects empty batches, batches above 1,000 rows, repeated feature rows, duplicate transaction IDs, and non-finite feature values;
- aggregates bounded row diagnostics and validates every row before making one model call, so malformed batches cannot produce partial predictions;
- constructs the model frame in the manifest's exact feature order regardless of mapping insertion order;
- requires `predict_proba` output with a fraud-probability column in `[0, 1]`;
- returns deterministic `schema_version: 1` output with explicit input/scored/failed row counts and ordered predictions containing calibrated fraud probability, `review`/`approve` decision, threshold, model version, transaction ID when supplied, and reason metadata.

## API runtime

`fraudshield.api:create_app` builds a FastAPI app around one local bundle. It exposes:

- `GET /health` for process liveness;
- `GET /ready` for loaded-model readiness;
- `GET /model-info` for public model metadata;
- `POST /score` for one transaction with Pydantic request validation;
- `POST /score-batch` for bounded batch scoring with Pydantic request validation.

Invalid payloads and scoring-contract violations return validation errors without tracebacks or secret-bearing diagnostics.

## Monitoring foundation

The monitoring layer compares a reference scoring window with a current scoring window using population stability index (PSI). `fraudshield.monitoring` learns fixed numeric buckets from the reference window, clips out-of-range current values into the edge buckets, smooths zero-count buckets, and classifies each feature or prediction-score signal against configurable warning and critical thresholds.

The CLI entry point:

```bash
uv run fraudshield monitor-drift reference.csv current.csv \
  --feature-column Time \
  --feature-column Amount \
  --score-column fraud_probability \
  --report-path reports/monitoring_report.json
```

writes versioned JSON evidence with row counts for both windows, PSI buckets, alert status, and responsible-use limitations. PSI is useful for operational distribution monitoring, but it does not replace labeled performance monitoring, fraud-review feedback loops, or governance checks.

## Offline and container paths

The offline CLI command:

```bash
uv run fraudshield score-batch input.csv \
  --model-bundle artifacts/model-bundle \
  --output-json reports/predictions.json
```

writes `schema_version`, `model_version`, all-or-nothing row accounting, and ordered prediction records for feature-only CSV input. The CLI reads at most 1,001 rows to enforce the 1,000-row bound, requires the exact ordered model-feature header, rejects repeated rows, and atomically replaces the JSON destination only after validation and scoring succeed.

The Docker runtime installs only the package, runs `uvicorn fraudshield.api:app`, uses a non-root `appuser`, and expects `FRAUDSHIELD_MODEL_BUNDLE` to point to a mounted/generated bundle directory. `.dockerignore` excludes local data, virtual environments, caches, and model artifacts from build context.

## Monitoring contract

`fraudshield.monitoring` produces versioned JSON reports for reference/current windows. Each report uses population stability index (PSI) for numeric feature and prediction-score distributions, learns stable bucket edges from the reference window only, clips current underflow/overflow values into the edge buckets, and smooths zero-count buckets so alerts remain finite instead of failing open.

`fraudshield monitor-drift` exposes the same contract from the CLI. Analysts can repeat `--feature-column` for the feature set and use `--score-column` for the calibrated fraud-probability field. Warning and critical PSI thresholds are explicit command parameters, and the report records row counts, per-bucket counts/proportions, PSI components, alert statuses, and limitations. The checked-in `reports/monitoring_report.json` compares the calibration window to the final holdout as an evidence-backed backtest example; production use would compare governed reference windows with current scored traffic and should pair drift alerts with current labels when they become available.
