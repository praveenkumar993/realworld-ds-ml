"""
main.py — FastAPI application for the Insurance Claim Cost Prediction API.

Endpoints
---------
GET  /health          Model version, training metrics, feature count
POST /quote           Single-policy prediction with SHAP explanation
POST /quote/batch     Batch prediction (1-1000 policies)
GET  /explain         GLM rating relativities (IRDAI-style table)

Usage
-----
  cd api/
  uvicorn main:app --reload --port 8000

Interactive docs at: http://localhost:8000/docs
"""

from __future__ import annotations
from contextlib import asynccontextmanager
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from schema import (
    QuoteRequest, QuoteResponse,
    BatchQuoteRequest, BatchQuoteResponse,
    HealthResponse, ExplainResponse,
)
from model import load_models, predict_one, predict_batch, get_relativities

_SUMMARY_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "model_summary.json"
)


# ── Lifespan: load models once at startup ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all artifacts before the first request arrives."""
    t0 = time.time()
    load_models()
    elapsed = time.time() - t0
    print(f"[startup] Models loaded in {elapsed:.2f}s")
    yield
    print("[shutdown] API shutting down")


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Insurance Claim Cost Prediction API",
    description = (
        "Motor insurance expected claim cost prediction "
        "using a Tweedie GLM (primary) and LightGBM (secondary). "
        "Trained on simulated Indian motor insurance data. "
        "See /docs for the full schema and example requests.\n\n"
        "⚠️ The training data is simulated. Known limitations apply "
        "(see /health for details). Not intended for production use."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Error handler ──────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


# ── Routes ────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and model metadata",
)
def health() -> HealthResponse:
    """
    Returns model version, Tweedie power, test-set Gini scores and
    the six confirmed data limitations.
    """
    m   = load_models()
    sb  = {r["model"]: r for r in m.summary["scoreboard"]}

    training_metrics = {
        "tweedie_power"            : m.summary["tweedie_power"],
        "n_trees_tweedie_lgbm"     : m.summary["n_trees"],
        "glm_alpha"                : m.summary["glm_alpha"],
        "test_norm_gini_glm"       : sb.get("Tweedie GLM", {}).get("norm_gini"),
        "test_norm_gini_lgbm"      : sb.get("Tweedie LightGBM", {}).get("norm_gini"),
        "test_norm_gini_premium"   : sb.get("Premium (scaled)", {}).get("norm_gini"),
        "frequency_auc"            : m.summary["frequency_auc"]["model"],
        "oracle_frequency_auc"     : m.summary["frequency_auc"]["oracle"],
        "known_limitations": [
            "Loss ratio ~407% (severity calibration artifact in training data)",
            "Monsoon effect present in severity only, not frequency",
            "Third-party severity multiplier unused in generator",
            "74.9% of TP-only policies carry own-damage claims in training data",
            "TP-only policies charged own-damage premium in training data",
            "Missing vehicle age inflated IDV 1.55x for 2% of training rows",
        ],
    }

    from features import FEATURES
    return HealthResponse(
        status          = "ok",
        model_version   = "1.0.0",
        tweedie_power   = m.summary["tweedie_power"],
        training_metrics= training_metrics,
        features_count  = len(FEATURES),
    )


@app.post(
    "/quote",
    response_model=QuoteResponse,
    summary="Predict expected claim cost for a single policy",
)
def quote(req: QuoteRequest) -> QuoteResponse:
    """
    Returns:
    - **expected_claim_cost_inr**: GLM prediction (primary model)
    - **model_premium_inr**: suggested premium = expected cost × 1.20 loading
    - **lgbm_expected_cost_inr**: LightGBM secondary estimate
    - **top_factors**: top 5 SHAP explanations from the LightGBM model

    Gender is not used by the served model even if supplied.
    """
    try:
        return predict_one(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/quote/batch",
    response_model=BatchQuoteResponse,
    summary="Predict expected claim cost for multiple policies",
)
def quote_batch(body: BatchQuoteRequest) -> BatchQuoteResponse:
    """
    Accepts 1-1000 policies in a single request. All SHAP values are
    computed in one batch call for efficiency.
    """
    try:
        preds = predict_batch(body.policies)
        return BatchQuoteResponse(predictions=preds, count=len(preds))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/explain",
    response_model=ExplainResponse,
    summary="GLM rating relativities (IRDAI-style table)",
)
def explain() -> ExplainResponse:
    """
    Returns the top 20 GLM multiplicative relativities.
    A relativity of 1.143 for vehicle_type_taxi means a taxi is expected
    to cost 1.143x the base rate, all else equal.
    Gender relativities are excluded from the served model.
    """
    m    = load_models()
    rels = get_relativities(top_n=20)
    return ExplainResponse(
        model            = "Tweedie GLM (log link, L2 penalty)",
        power            = m.summary["tweedie_power"],
        top_relativities = rels,
        note             = (
            "Relativities are multiplicative on expected claim cost. "
            "One-hot features within the same category should be compared "
            "to each other, not to the intercept. "
            "Gender is excluded from the served model."
        ),
    )