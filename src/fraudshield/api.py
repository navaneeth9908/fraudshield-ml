"""FastAPI application for FraudShield production inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from fraudshield.serving import (
    BATCH_MAX_ROWS,
    BATCH_SCORING_SCHEMA_VERSION,
    FraudPrediction,
    LoadedModelBundle,
    load_model_bundle,
)

DEFAULT_MODEL_BUNDLE_ENV = "FRAUDSHIELD_MODEL_BUNDLE"


class ScoreRequest(BaseModel):
    """Single-transaction scoring request."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = None
    features: dict[str, float]


class BatchScoreRequest(BaseModel):
    """Bounded batch-scoring request."""

    model_config = ConfigDict(extra="forbid")

    transactions: Annotated[list[ScoreRequest], Field(min_length=1, max_length=BATCH_MAX_ROWS)]


class ScoreResponse(BaseModel):
    """Single public scoring response."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = None
    fraud_probability: float
    decision: str
    threshold: float
    reason: str
    model_version: str


class BatchScoreSummary(BaseModel):
    """Fail-closed row accounting for a successful batch."""

    model_config = ConfigDict(extra="forbid")

    input_rows: int
    scored_rows: int
    failed_rows: Literal[0]


class BatchScoreResponse(BaseModel):
    """Batch public scoring response."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    model_version: str
    batch: BatchScoreSummary
    predictions: list[ScoreResponse]


def create_app(model_bundle: Path | None = None) -> FastAPI:
    """Create an API app, loading a local model bundle when configured."""
    app = FastAPI(
        title="FraudShield ML Inference API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )
    bundle: LoadedModelBundle | None = None
    load_error: str | None = None
    if model_bundle is not None:
        try:
            bundle = load_model_bundle(model_bundle)
        except (OSError, ValueError) as exc:
            load_error = str(exc)

    def require_bundle() -> LoadedModelBundle:
        if bundle is None:
            detail = "model bundle is not loaded"
            if load_error:
                detail = f"{detail}: {load_error}"
            raise HTTPException(status_code=503, detail=detail)
        return bundle

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        if bundle is None:
            raise HTTPException(status_code=503, detail=load_error or "model bundle is not loaded")
        return {"model_loaded": True, "model_version": bundle.metadata.model_version}

    @app.get("/model-info")
    def model_info() -> dict[str, object]:
        return require_bundle().model_info()

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> dict[str, object]:
        prediction = _score_safely(
            require_bundle(), request.features, transaction_id=request.transaction_id
        )
        return prediction.to_dict()

    @app.post("/score-batch", response_model=BatchScoreResponse)
    def score_batch(request: BatchScoreRequest) -> dict[str, object]:
        current_bundle = require_bundle()
        try:
            predictions = current_bundle.score_batch(
                [transaction.features for transaction in request.transactions],
                transaction_ids=[
                    transaction.transaction_id for transaction in request.transactions
                ],
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {
            "schema_version": BATCH_SCORING_SCHEMA_VERSION,
            "model_version": current_bundle.metadata.model_version,
            "batch": {
                "input_rows": len(request.transactions),
                "scored_rows": len(predictions),
                "failed_rows": 0,
            },
            "predictions": [prediction.to_dict() for prediction in predictions],
        }

    return app


def create_app_from_env() -> FastAPI:
    """Create the default ASGI app using an environment-provided bundle path."""
    bundle_path = os.environ.get(DEFAULT_MODEL_BUNDLE_ENV)
    return create_app(Path(bundle_path) if bundle_path else None)


def _score_safely(
    bundle: LoadedModelBundle, features: dict[str, float], *, transaction_id: str | None
) -> FraudPrediction:
    try:
        return bundle.score_one(features, transaction_id=transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


app = create_app_from_env()
