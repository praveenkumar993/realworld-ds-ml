"""
features.py — Pure-Python feature builder that mirrors Notebook 04.

Every transformation here must produce the same numeric value as the
corresponding Spark / pandas operation in NB04. The parity test in
tests/test_parity.py verifies this on 200 rows from modeling_data.csv.

No Spark dependency. No database dependency.
Input : a QuoteRequest (validated Pydantic model)
Output: a dict[str, float] of the 66 features in FEATURES order
"""

from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from schema import QuoteRequest


# ── Load the feature spec once at import time ─────────────────
_SPEC_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "feature_spec.json"
with open(_SPEC_PATH) as f:
    SPEC = json.load(f)

FEATURES       : list[str]       = SPEC["feature_cols"]
NUMERIC        : list[str]       = SPEC["numeric"]
CATEGORICAL    : list[str]       = SPEC["categorical"]
CATEGORY_LEVELS: dict[str, list] = SPEC["category_levels"]

# Medians for imputation — computed from modeling_data.csv at startup
# (populated by load_imputation_stats() below)
_IMPUTE: dict[str, float] = {}


def load_imputation_stats(modeling_csv: Path) -> None:
    """
    Compute and cache the column medians used for imputation.
    Called once from model.py at startup.
    We only need medians for numeric features that can be missing in
    practice; currently none are missing after NB02 cleaning, but
    this guards against future data gaps.
    """
    df = pd.read_csv(modeling_csv, usecols=NUMERIC)
    for col in NUMERIC:
        if col in df.columns:
            _IMPUTE[col] = float(df[col].median())


def _safe(value: Any, fallback: float) -> float:
    """Return fallback if value is None, NaN or infinite."""
    if value is None:
        return fallback
    try:
        f = float(value)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return fallback


def build_features(req: QuoteRequest) -> np.ndarray:
    """
    Turn a validated QuoteRequest into the 66-element feature vector
    expected by the trained models.

    Returns a 1-D numpy array of shape (66,) in FEATURES order.
    Raises ValueError if a required numeric field is missing after
    imputation (should not happen given Pydantic validation).
    """
    seg = req.vehicle_segment   # derived property on the schema

    # ── Numeric features ──────────────────────────────────────
    log_income = math.log1p(_safe(req.monthly_income,
                                  _IMPUTE.get("monthly_income", 50000)))
    log_idv    = math.log(_safe(req.idv, _IMPUTE.get("idv", 500000)))

    age      = _safe(req.age, _IMPUTE.get("age", 35))
    exp      = _safe(req.driving_exp_years,
                     _IMPUTE.get("driving_exp_years", 5))
    # experience_ratio: share of adult life spent driving, clipped to 1
    experience_ratio = min(1.0, exp / max(age - 17.0, 1.0))

    idv_to_annual_income = _safe(req.idv, 500000) / (
        _safe(req.monthly_income, 50000) * 12 + 1.0
    )

    has_od_cover = int(req.policy_type != "third_party_only")

    numeric_vals: dict[str, float] = {
        "age"                   : age,
        "driving_exp_years"     : exp,
        "log_income"            : log_income,
        "prior_claims_count"    : float(req.prior_claims_count),
        "ncb_percentage"        : float(req.ncb_percentage),
        "is_young_driver"       : float(age < 25),
        "is_senior_driver"      : float(age >= 60),
        "experience_ratio"      : experience_ratio,
        "has_prior_claim"       : float(req.prior_claims_count > 0),
        "vehicle_age_years"     : _safe(req.vehicle_age_years,
                                        _IMPUTE.get("vehicle_age_years", 4)),
        "engine_cc"             : float(req.engine_cc),
        "log_idv"               : log_idv,
        "is_modified"           : float(req.is_modified),
        "is_new_vehicle"        : float(req.vehicle_age_years <= 1),
        "is_old_vehicle"        : float(req.vehicle_age_years >= 8),
        "idv_to_annual_income"  : round(idv_to_annual_income, 4),
        "voluntary_deductible"  : float(req.voluntary_deductible),
        "has_deductible"        : float(req.voluntary_deductible > 0),
        "addon_count"           : float(req.addon_count),
        "has_zero_depreciation" : float(req.has_zero_depreciation),
        "has_engine_protection" : float(req.has_engine_protection),
        "has_roadside_assistance": float(req.has_roadside_assistance),
        "has_consumables"       : float(req.has_consumables),
        "has_od_cover"          : float(has_od_cover),
        "policy_start_quarter"  : float(req.policy_start_quarter),
    }

    # ── One-hot features ──────────────────────────────────────
    cat_inputs: dict[str, str] = {
        "vehicle_make"    : req.vehicle_make,
        "vehicle_segment" : seg,
        "vehicle_type"    : req.vehicle_type,
        "fuel_type"       : req.fuel_type,
        "policy_type"     : req.policy_type,
        "city_tier"       : req.city_tier,
        "occupation"      : req.occupation,
        "gender"          : "Not Specified",   # gender dropped from serving
    }

    one_hot: dict[str, float] = {}
    for cat, value in cat_inputs.items():
        levels = CATEGORY_LEVELS[cat]
        for level in levels:
            col = f"{cat}_{level}"
            one_hot[col] = float(value == level)

    # ── Assemble in FEATURES order ────────────────────────────
    combined = {**numeric_vals, **one_hot}
    result   = []
    missing  = []
    for col in FEATURES:
        if col in combined:
            result.append(combined[col])
        else:
            # This should never happen if schema and spec stay in sync
            missing.append(col)
            result.append(0.0)

    if missing:
        raise ValueError(
            f"build_features: {len(missing)} feature(s) not computed: {missing}. "
            "The feature spec may be out of sync with features.py."
        )

    return np.array(result, dtype=np.float64)


def build_feature_df(req: QuoteRequest) -> pd.DataFrame:
    """Wrap build_features in a single-row DataFrame for sklearn pipelines."""
    return pd.DataFrame([build_features(req)], columns=FEATURES)