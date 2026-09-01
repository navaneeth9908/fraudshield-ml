"""Versioned model-bundle loading and typed prediction service primitives."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraudshield.data import FEATURE_COLUMNS, file_sha256

MODEL_BUNDLE_SCHEMA_VERSION = 1
BATCH_SCORE_SCHEMA_VERSION = 1
MAX_BATCH_SIZE = 1_000
BATCH_SCORING_SCHEMA_VERSION = BATCH_SCORE_SCHEMA_VERSION
BATCH_MAX_ROWS = MAX_BATCH_SIZE
MODEL_ARTIFACT_NAME = "model.joblib"
MODEL_MANIFEST_NAME = "model-bundle.json"


class ProbabilityModel(Protocol):
    """Minimal protocol for fitted binary fraud probability models."""

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        """Return genuine/fraud probabilities for each row."""


class BatchValidationError(ValueError):
    """Fail-closed batch validation error with a versioned, bounded summary."""

    schema_version = BATCH_SCORING_SCHEMA_VERSION

    def __init__(self, message: str, *, issues: Sequence[str] = ()) -> None:
        self.issues = tuple(issues)
        self.issue_count = len(self.issues)
        if self.issues:
            detail = "; ".join(self.issues)
            message = f"{message} with {self.issue_count} issues: {detail}"
        super().__init__(message)


BatchScoringError = BatchValidationError


@dataclass(frozen=True)
class ModelBundleMetadata:
    """Human-readable and operational metadata persisted with a model bundle."""

    model_name: str
    model_version: str
    decision_threshold: float
    decision_policy: str
    training_data_sha256: str
    calibration_method: str = "sigmoid"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if not 0.0 <= self.decision_threshold <= 1.0 or not isfinite(self.decision_threshold):
            raise ValueError("decision_threshold must be a finite probability between 0 and 1")
        if not self.decision_policy.strip():
            raise ValueError("decision_policy must not be empty")
        if len(self.training_data_sha256) != 64:
            raise ValueError("training_data_sha256 must be a 64-character SHA-256 digest")


@dataclass(frozen=True)
class FraudPrediction:
    """Public prediction response for a scored transaction."""

    transaction_id: str | None
    fraud_probability: float
    decision: str
    threshold: float
    reason: str
    model_version: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable prediction output."""
        payload: dict[str, object] = {
            "fraud_probability": self.fraud_probability,
            "decision": self.decision,
            "threshold": self.threshold,
            "reason": self.reason,
            "model_version": self.model_version,
        }
        if self.transaction_id is not None:
            payload["transaction_id"] = self.transaction_id
        return payload


@dataclass(frozen=True)
class BatchScoringOutput:
    """Versioned all-or-nothing batch output with explicit row accounting."""

    model_version: str
    predictions: tuple[FraudPrediction, ...]
    schema_version: int = BATCH_SCORE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable batch response."""
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "batch": {
                "input_rows": len(self.predictions),
                "scored_rows": len(self.predictions),
                "failed_rows": 0,
            },
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True)
class LoadedModelBundle:
    """Validated model artifact plus its versioned serving contract."""

    model: ProbabilityModel
    metadata: ModelBundleMetadata
    manifest: dict[str, object]
    bundle_dir: Path

    def score_one(
        self, features: Mapping[str, float], *, transaction_id: str | None = None
    ) -> FraudPrediction:
        """Score one feature mapping and attach decision metadata."""
        return self.score_batch([features], transaction_ids=[transaction_id])[0]

    def score_batch(
        self,
        transactions: Sequence[Mapping[str, float]],
        *,
        transaction_ids: Sequence[str | None] | None = None,
    ) -> tuple[FraudPrediction, ...]:
        """Score a batch of feature mappings while preserving row order."""
        if len(transactions) == 0:
            raise BatchScoringError("transactions must not be empty")
        if len(transactions) > BATCH_MAX_ROWS:
            raise BatchScoringError(
                f"batch must contain at most {BATCH_MAX_ROWS} transactions "
                f"(received {len(transactions)})"
            )
        if transaction_ids is not None and len(transaction_ids) != len(transactions):
            raise BatchScoringError("transaction_ids must match transactions length")
        ids = transaction_ids if transaction_ids is not None else tuple(None for _ in transactions)
        frame = _validated_feature_frame(transactions, transaction_ids=ids)
        probabilities = _fraud_probabilities(self.model, frame)
        predictions: list[FraudPrediction] = []
        for transaction_id, probability in zip(ids, probabilities, strict=True):
            decision = "review" if probability >= self.metadata.decision_threshold else "approve"
            comparison = "at_or_above" if decision == "review" else "below"
            reason = (
                f"fraud_probability {comparison} {self.metadata.decision_threshold:.4f} "
                f"for {self.metadata.model_name} {self.metadata.model_version}"
            )
            predictions.append(
                FraudPrediction(
                    transaction_id=transaction_id,
                    fraud_probability=probability,
                    decision=decision,
                    threshold=self.metadata.decision_threshold,
                    reason=reason,
                    model_version=self.metadata.model_version,
                )
            )
        return tuple(predictions)

    def score_batch_output(
        self,
        transactions: Sequence[Mapping[str, float]],
        *,
        transaction_ids: Sequence[str | None] | None = None,
    ) -> BatchScoringOutput:
        """Score a batch into the versioned public all-or-nothing contract."""
        predictions = self.score_batch(transactions, transaction_ids=transaction_ids)
        return BatchScoringOutput(
            model_version=self.metadata.model_version,
            predictions=predictions,
        )

    def model_info(self) -> dict[str, object]:
        """Return compact public model metadata without exposing artifact internals."""
        return {
            "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
            "model_name": self.metadata.model_name,
            "model_version": self.metadata.model_version,
            "feature_count": len(FEATURE_COLUMNS),
            "decision_threshold": self.metadata.decision_threshold,
            "decision_policy": self.metadata.decision_policy,
            "calibration_method": self.metadata.calibration_method,
        }


def write_model_bundle(
    *, model: ProbabilityModel, bundle_dir: Path, metadata: ModelBundleMetadata
) -> Path:
    """Persist a fitted model with a compact versioned manifest."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = bundle_dir / MODEL_ARTIFACT_NAME
    joblib.dump(model, artifact_path)
    manifest = _manifest_payload(metadata, artifact_sha256=file_sha256(artifact_path))
    manifest_path = bundle_dir / MODEL_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def load_model_bundle(bundle_dir: Path) -> LoadedModelBundle:
    """Load and validate a model bundle from a directory."""
    manifest_path = bundle_dir / MODEL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"model bundle manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("model bundle manifest must be a JSON object")
    _validate_manifest(manifest)

    model_section = cast(dict[str, object], manifest["model"])
    artifact_name = cast(str, model_section["artifact_path"])
    artifact_path = _safe_artifact_path(bundle_dir, artifact_name)
    expected_sha256 = cast(str, model_section["artifact_sha256"])
    actual_sha256 = file_sha256(artifact_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("model artifact SHA-256 does not match the bundle manifest")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
            module="joblib.numpy_pickle",
        )
        loaded = joblib.load(artifact_path)
    model = cast(ProbabilityModel, loaded)
    decision_section = cast(dict[str, object], manifest["decision"])
    training_data_section = cast(dict[str, object], manifest["training_data"])
    calibration_section = cast(dict[str, object], manifest["calibration"])
    metadata = ModelBundleMetadata(
        model_name=cast(str, model_section["name"]),
        model_version=cast(str, model_section["version"]),
        decision_threshold=cast(float, decision_section["threshold"]),
        decision_policy=cast(str, decision_section["policy"]),
        training_data_sha256=cast(str, training_data_section["sha256"]),
        calibration_method=cast(str, calibration_section["method"]),
    )
    return LoadedModelBundle(
        model=model, metadata=metadata, manifest=manifest, bundle_dir=bundle_dir
    )


def _manifest_payload(metadata: ModelBundleMetadata, *, artifact_sha256: str) -> dict[str, object]:
    return {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "model": {
            "name": metadata.model_name,
            "version": metadata.model_version,
            "artifact_path": MODEL_ARTIFACT_NAME,
            "artifact_sha256": artifact_sha256,
            "format": "joblib",
        },
        "features": list(FEATURE_COLUMNS),
        "decision": {
            "threshold": metadata.decision_threshold,
            "policy": metadata.decision_policy,
        },
        "calibration": {"method": metadata.calibration_method, "probability": "fraud"},
        "training_data": {"sha256": metadata.training_data_sha256},
    }


def _validate_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("schema_version") != MODEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported model bundle schema_version")
    features = manifest.get("features")
    if features != list(FEATURE_COLUMNS):
        raise ValueError("model bundle features do not match the scoring contract")
    for section in ("model", "decision", "calibration", "training_data"):
        if not isinstance(manifest.get(section), dict):
            raise ValueError(f"model bundle section {section!r} must be an object")
    model_section = cast(dict[str, object], manifest["model"])
    for key in ("name", "version", "artifact_path", "artifact_sha256", "format"):
        if not isinstance(model_section.get(key), str):
            raise ValueError(f"model bundle model.{key} must be a string")
    if model_section["format"] != "joblib":
        raise ValueError("only joblib model artifacts are supported")
    decision_section = cast(dict[str, object], manifest["decision"])
    threshold = decision_section.get("threshold")
    if not isinstance(threshold, int | float):
        raise ValueError("model bundle decision.threshold must be numeric")
    if not isinstance(decision_section.get("policy"), str):
        raise ValueError("model bundle decision.policy must be a string")
    calibration_section = cast(dict[str, object], manifest["calibration"])
    if not isinstance(calibration_section.get("method"), str):
        raise ValueError("model bundle calibration.method must be a string")
    training_data_section = cast(dict[str, object], manifest["training_data"])
    if not isinstance(training_data_section.get("sha256"), str):
        raise ValueError("model bundle training_data.sha256 must be a string")


def _safe_artifact_path(bundle_dir: Path, artifact_name: str) -> Path:
    artifact_path = Path(artifact_name)
    if artifact_path.is_absolute() or artifact_path.name != artifact_name:
        raise ValueError("model artifact_path must be a filename inside the bundle directory")
    resolved = bundle_dir / artifact_path
    if not resolved.is_file():
        raise ValueError(f"model artifact does not exist: {resolved}")
    return resolved


def _validated_feature_frame(
    transactions: Sequence[Mapping[str, float]],
    *,
    transaction_ids: Sequence[str | None],
) -> pd.DataFrame:
    issues: list[str] = []
    first_id_row: dict[str, int] = {}
    for index, transaction_id in enumerate(transaction_ids):
        if transaction_id is None:
            continue
        if not transaction_id.strip():
            issues.append(f"row {index} transaction_id must not be blank")
            continue
        first_index = first_id_row.setdefault(transaction_id, index)
        if first_index != index:
            issues.append(
                f"row {index} duplicates transaction_id {transaction_id!r} from row {first_index} "
                f"(duplicate transaction_id {transaction_id!r} at rows {first_index}, {index})"
            )

    rows: list[dict[str, float]] = []
    fingerprints: dict[tuple[float, ...], list[int]] = {}
    expected = set(FEATURE_COLUMNS)
    for index, transaction in enumerate(transactions):
        keys = set(transaction.keys())
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing:
            issues.append(f"row {index} missing required model features: {', '.join(missing)}")
        if extra:
            issues.append(f"row {index} contains unsupported model features: {', '.join(extra)}")
        values: dict[str, float] = {}
        for column in FEATURE_COLUMNS:
            if column not in transaction:
                continue
            raw_value = transaction[column]
            if isinstance(raw_value, bool):
                issues.append(f"row {index} feature {column} must be numeric")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                issues.append(f"row {index} feature {column} must be numeric")
                continue
            if not isfinite(value):
                issues.append(f"row {index} feature {column} must be finite")
                continue
            values[column] = value
        if len(values) == len(FEATURE_COLUMNS) and not extra:
            fingerprint = tuple(values[column] for column in FEATURE_COLUMNS)
            fingerprints.setdefault(fingerprint, []).append(index)
            rows.append(values)
    for indexes in fingerprints.values():
        first_index = indexes[0]
        for index in indexes[1:]:
            issues.append(
                f"row {index} duplicates row {first_index} "
                f"(duplicate transaction features at rows {first_index}, {index})"
            )
    if issues:
        raise BatchScoringError("batch validation failed", issues=issues)
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


def _fraud_probabilities(model: ProbabilityModel, frame: pd.DataFrame) -> tuple[float, ...]:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(frame)
        or probabilities.shape[1] < 2
    ):
        raise ValueError("model predict_proba must return an n-by-2 probability matrix")
    fraud_scores = probabilities[:, 1]
    if not np.isfinite(fraud_scores).all():
        raise ValueError("model returned non-finite fraud probabilities")
    if ((fraud_scores < 0.0) | (fraud_scores > 1.0)).any():
        raise ValueError("model returned probabilities outside [0, 1]")
    return tuple(float(score) for score in fraud_scores.tolist())
