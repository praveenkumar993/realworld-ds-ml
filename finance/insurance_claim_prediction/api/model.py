"""
model.py — Load all trained artifacts once at startup and serve predictions.

Models loaded:
  glm  : Tweedie GLM  (sklearn Pipeline: StandardScaler + TweedieRegressor)
  gbm  : Tweedie LightGBM
  freq : LightGBM frequency classifier
  sev  : LightGBM severity (gamma, on claimants only)

The GLM is the primary model for the /quote endpoint.
LightGBM is a secondary estimate included in the response.
SHAP values come from the LightGBM explainer (faster than GLM SHAP).

Gender is NOT used by the served model. The GLM was trained with gender;
we zero out the gender columns in the feature vector before calling it.
The Gini difference is < 0.001 on this dataset (see NB06).
"""

from __future__ import annotations
import json
import math
import pickle
import numpy as np
import shap
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from features import (
    FEATURES, CATEGORY_LEVELS, build_features,
    load_imputation_stats,
)
from schema import QuoteRequest, QuoteResponse, ShapExplanation

# ── Paths ──────────────────────────────────────────────────────
_BASE       = Path(__file__).resolve().parent.parent / "data" / "processed"
_MODELS_PKL = _BASE / "models.pkl"
_SPEC_JSON  = _BASE / "feature_spec.json"
_SUMMARY    = _BASE / "model_summary.json"
_MODELING   = _BASE / "modeling_data.csv"

Y_SCALE     = 1000.0   # models were trained on claim_amount / 1000
LOADING     = 1.20     # 20% expense + profit margin


class LoadedModels(NamedTuple):
    glm         : object   # sklearn Pipeline
    gbm         : object   # LGBMRegressor
    freq        : object   # LGBMClassifier
    sev         : object   # LGBMRegressor (severity ratio)
    shap_explainer: shap.TreeExplainer
    summary     : dict
    gender_cols : list[str]   # columns to zero out in feature vector


@lru_cache(maxsize=1)
def load_models() -> LoadedModels:
    """
    Load all artifacts from disk. lru_cache ensures this runs once.
    Called from the FastAPI lifespan event so it happens at startup,
    not on the first request.
    """
    # Imputation stats (medians) for the feature builder
    load_imputation_stats(_MODELING)

    with open(_MODELS_PKL, "rb") as f:
        mdl = pickle.load(f)

    with open(_SUMMARY) as f:
        summary = json.load(f)

    gbm_model = mdl["gbm"]
    explainer = shap.TreeExplainer(gbm_model)

    # Identify gender columns so we can zero them out
    gender_cols = [c for c in FEATURES if c.startswith("gender_")]

    return LoadedModels(
        glm=mdl["glm"],
        gbm=gbm_model,
        freq=mdl["freq"],
        sev=mdl["sev"],
        shap_explainer=explainer,
        summary=summary,
        gender_cols=gender_cols,
    )


def _zero_gender(vec: np.ndarray, gender_cols: list[str]) -> np.ndarray:
    """Return a copy of vec with gender one-hot columns zeroed out."""
    v = vec.copy()
    for col in gender_cols:
        if col in FEATURES:
            v[FEATURES.index(col)] = 0.0
    return v


def predict_one(req: QuoteRequest) -> QuoteResponse:
    """
    Full prediction pipeline for a single policy.

    Steps:
    1. Build the 66-feature vector from the raw request
    2. Zero out gender columns
    3. GLM prediction (primary)
    4. LightGBM Tweedie prediction (secondary)
    5. SHAP values from the LightGBM explainer
    6. Assemble and return QuoteResponse
    """
    m = load_models()

    raw_vec     = build_features(req)
    vec         = _zero_gender(raw_vec, m.gender_cols)
    glm_pred_k = float(m.glm.predict(vec.reshape(1, -1))[0])

    # GLM prediction (in ₹/1000, then scale back)
    glm_pred    = max(0.0, glm_pred_k * Y_SCALE)

    # LightGBM prediction
    gbm_pred_k  = float(m.gbm.predict(vec.reshape(1, -1))[0])
    gbm_pred    = max(0.0, gbm_pred_k * Y_SCALE)

    # SHAP on the LightGBM model
    sv          = m.shap_explainer.shap_values(vec.reshape(1, -1))[0]
    shap_abs    = np.abs(sv)
    top_idx     = np.argsort(shap_abs)[::-1][:5]

    top_factors = []
    for i in top_idx:
        top_factors.append(ShapExplanation(
            feature           = FEATURES[i],
            value             = float(vec[i]),
            shap_contribution = float(sv[i]),
            direction         = "increases_cost" if sv[i] > 0 else "decreases_cost",
        ))

    warning = None
    if glm_pred > req.idv * 0.8:
        warning = (
            "Predicted cost exceeds 80% of IDV. "
            "Verify IDV and vehicle details."
        )

    return QuoteResponse(
        expected_claim_cost_inr = round(glm_pred, 2),
        model_premium_inr       = round(glm_pred * LOADING, 2),
        lgbm_expected_cost_inr  = round(gbm_pred, 2),
        top_factors             = top_factors,
        warning                 = warning,
    )


def predict_batch(reqs: list[QuoteRequest]) -> list[QuoteResponse]:
    """
    Batch prediction. Builds one feature matrix and calls models once.
    SHAP is computed row-by-row because shap_values accepts 2-D arrays
    but we still need per-row top-5 explanations.
    """
    m   = load_models()
    n   = len(reqs)
    mat = np.zeros((n, len(FEATURES)), dtype=np.float64)
    for i, req in enumerate(reqs):
        raw = build_features(req)
        mat[i] = _zero_gender(raw, m.gender_cols)
    glm_preds = np.maximum(m.glm.predict(mat) * Y_SCALE, 0.0)
    gbm_preds = np.maximum(m.gbm.predict(mat)   * Y_SCALE, 0.0)
    sv_all    = m.shap_explainer.shap_values(mat)   # (n, 66)

    results = []
    for i, req in enumerate(reqs):
        sv      = sv_all[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:5]
        factors = [
            ShapExplanation(
                feature           = FEATURES[j],
                value             = float(mat[i, j]),
                shap_contribution = float(sv[j]),
                direction         = "increases_cost" if sv[j] > 0
                                    else "decreases_cost",
            )
            for j in top_idx
        ]
        glm_p = float(glm_preds[i])
        warning = None
        if glm_p > req.idv * 0.8:
            warning = "Predicted cost exceeds 80% of IDV. Verify details."

        results.append(QuoteResponse(
            expected_claim_cost_inr = round(glm_p, 2),
            model_premium_inr       = round(glm_p * LOADING, 2),
            lgbm_expected_cost_inr  = round(float(gbm_preds[i]), 2),
            top_factors             = factors,
            warning                 = warning,
        ))
    return results


def get_relativities(top_n: int = 20) -> list[dict]:
    """
    Return the top-N GLM rating relativities sorted by magnitude.
    These are exp(coef) for each feature, representing the multiplicative
    effect on expected claim cost.
    """
    m      = load_models()
    glm    = m.glm
    scaler = glm[0]
    coefs  = glm[-1].coef_

    rels = []
    for feat, coef in zip(FEATURES, coefs):
        if feat.startswith("gender_"):
            continue                    # gender excluded from serving
        rel = math.exp(float(coef))
        if abs(rel - 1.0) > 0.01:      # only non-trivial effects
            rels.append({
                "feature"    : feat,
                "relativity" : round(rel, 4),
                "direction"  : "more_expensive" if rel > 1 else "cheaper",
            })
    rels.sort(key=lambda r: abs(r["relativity"] - 1.0), reverse=True)
    return rels[:top_n]