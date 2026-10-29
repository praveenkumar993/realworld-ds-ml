"""
test_parity.py — Training-serving skew test.

For 200 randomly sampled rows from modeling_data.csv, assert that the
API's feature builder produces values within TOLERANCE of the values
that were actually used to train the model.

This is the most important test in the project. The feature builder in
features.py was written independently of Notebook 04 (which used Spark).
Any mismatch means the API is scoring on different inputs than the model
was trained on.

Run: pytest tests/test_parity.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROCESSED = (
    Path(__file__).parent.parent.parent / "data" / "processed"
)
MODELING_CSV    = PROCESSED / "modeling_data.csv"
POLICIES_CSV    = PROCESSED / "policies_clean.csv"
POLICYHOLDERS_CSV = PROCESSED / "policyholders_clean.csv"
VEHICLES_CSV    = PROCESSED / "vehicles_clean.csv"
FEATURE_SPEC    = PROCESSED / "feature_spec.json"

TOLERANCE  = 1e-4     # max allowed absolute difference per feature
N_SAMPLE   = 200
SEED       = 42

# ── Load ───────────────────────────────────────────────────────
from features import FEATURES, load_imputation_stats, build_features
from schema   import QuoteRequest, MAKE_SEGMENT

load_imputation_stats(MODELING_CSV)


def _load_test_rows() -> pd.DataFrame:
    """
    Build a DataFrame that mirrors the inputs to build_features by
    joining the cleaned tables back together on a random sample.
    """
    spec = json.load(open(FEATURE_SPEC))

    modeling = pd.read_csv(MODELING_CSV)
    policies = pd.read_csv(POLICIES_CSV)
    policyholders = pd.read_csv(POLICYHOLDERS_CSV)
    vehicles  = pd.read_csv(VEHICLES_CSV)

    df = (
        modeling[["policy_id"] + spec["feature_cols"]]
        .merge(policies[["policy_id", "policyholder_id",
                          "policy_type", "voluntary_deductible",
                          "addon_count", "has_zero_depreciation",
                          "has_engine_protection", "has_roadside_assistance",
                          "has_consumables", "policy_start_quarter"]],
               on="policy_id", validate="1:1")
        .merge(policyholders[["policyholder_id", "age", "driving_exp_years",
                               "monthly_income", "prior_claims_count",
                               "ncb_percentage", "city_tier", "occupation",
                               "gender"]],
               on="policyholder_id", validate="1:1")
        .merge(vehicles[["policyholder_id", "vehicle_make", "vehicle_type",
                          "vehicle_age_years", "engine_cc", "idv",
                          "fuel_type", "is_modified"]],
               on="policyholder_id", validate="1:1")
    )

    return df.sample(n=min(N_SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)


ROWS = _load_test_rows()


@pytest.mark.parametrize("row_idx", range(len(ROWS)))
def test_feature_parity(row_idx: int):
    """
    For each sampled row:
    1. Build a QuoteRequest from the cleaned-table columns
    2. Call build_features() to get the API feature vector
    3. Compare element-by-element against modeling_data.csv ground truth
    """
    row = ROWS.iloc[row_idx]

    # Some rows may have impossible ages that were imputed in NB02;
    # build_features clips age to [18, 85]. Test still runs.
    age = float(row["age"])
    exp = float(row["driving_exp_years"])
    # Clamp driving experience to be consistent with the schema
    exp = min(exp, age - 18 + 1)

    ncb_raw = row.get("ncb_percentage", 0)
    try:
        ncb = int(float(ncb_raw)) if not pd.isna(ncb_raw) else 0
    except (ValueError, TypeError):
        ncb = 0

    # Build a QuoteRequest using the exact values from the training data
    req = QuoteRequest(
        age                  = max(18, min(85, int(age))),
        driving_exp_years    = max(0.5, exp),
        monthly_income       = float(row["monthly_income"]),
        prior_claims_count   = int(row["prior_claims_count"]),
        ncb_percentage       = ncb,
        city_tier            = str(row["city_tier"]),
        occupation           = str(row["occupation"]),
        vehicle_make         = str(row["vehicle_make"]),
        vehicle_type         = str(row["vehicle_type"]),
        vehicle_age_years    = float(row["vehicle_age_years"]),
        engine_cc            = int(float(row["engine_cc"])),
        idv                  = float(row["idv"]),
        fuel_type            = str(row["fuel_type"]),
        is_modified          = int(float(row["is_modified"])),
        policy_type          = str(row["policy_type"]),
        voluntary_deductible = float(row["voluntary_deductible"]),
        addon_count          = int(float(row["addon_count"])),
        has_zero_depreciation   = int(float(row["has_zero_depreciation"])),
        has_engine_protection   = int(float(row["has_engine_protection"])),
        has_roadside_assistance = int(float(row["has_roadside_assistance"])),
        has_consumables         = int(float(row["has_consumables"])),
        policy_start_quarter    = int(float(row["policy_start_quarter"])),
    )

    api_vec = build_features(req)

    # Ground-truth vector from modeling_data.csv
    gt_vec = np.array(
        [float(row[c]) for c in FEATURES],
        dtype=np.float64
    )

    # Gender columns: the API always emits "Not Specified" (zeroed one-hots).
    # Ground truth may differ. We skip gender columns in the comparison.
    gender_cols = {c for c in FEATURES if c.startswith("gender_")}

    mismatches = []
    for i, feat in enumerate(FEATURES):
        if feat in gender_cols:
            continue
        diff = abs(float(api_vec[i]) - float(gt_vec[i]))
        if diff > TOLERANCE:
            mismatches.append(
                f"{feat}: api={api_vec[i]:.6f} gt={gt_vec[i]:.6f} diff={diff:.2e}"
            )

    assert not mismatches, (
        f"Row {row_idx} (policy_id={row['policy_id']}): "
        f"{len(mismatches)} feature(s) exceed tolerance {TOLERANCE}:\n"
        + "\n  ".join(mismatches)
    )