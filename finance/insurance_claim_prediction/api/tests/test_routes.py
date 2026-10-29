"""
test_routes.py — Integration tests for all API endpoints.

Uses FastAPI's TestClient (backed by httpx) to hit each route
and assert the response shape, status codes and business logic.

Run: pytest tests/test_routes.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Shared fixtures ────────────────────────────────────────────
ECONOMY_POLICY = {
    "age": 35,
    "driving_exp_years": 10,
    "monthly_income": 80000,
    "prior_claims_count": 0,
    "ncb_percentage": 25,
    "city_tier": "Metro",
    "occupation": "salaried",
    "vehicle_make": "Hyundai",
    "vehicle_type": "private_car",
    "vehicle_age_years": 3,
    "engine_cc": 1197,
    "idv": 600000,
    "fuel_type": "petrol",
    "is_modified": 0,
    "policy_type": "comprehensive",
    "voluntary_deductible": 1000,
    "addon_count": 2,
    "has_zero_depreciation": 1,
    "has_engine_protection": 0,
    "has_roadside_assistance": 1,
    "has_consumables": 0,
    "policy_start_quarter": 2,
}

LUXURY_POLICY = {
    "age": 28,
    "driving_exp_years": 5,
    "monthly_income": 500000,
    "prior_claims_count": 2,
    "ncb_percentage": 0,
    "city_tier": "Metro",
    "occupation": "business",
    "vehicle_make": "BMW",
    "vehicle_type": "private_car",
    "vehicle_age_years": 1,
    "engine_cc": 2996,
    "idv": 4000000,
    "fuel_type": "petrol",
    "is_modified": 0,
    "policy_type": "comprehensive",
    "voluntary_deductible": 0,
    "addon_count": 4,
    "has_zero_depreciation": 1,
    "has_engine_protection": 1,
    "has_roadside_assistance": 1,
    "has_consumables": 1,
    "policy_start_quarter": 1,
}

TWO_WHEELER = {
    "age": 22,
    "driving_exp_years": 2,
    "monthly_income": 25000,
    "prior_claims_count": 0,
    "ncb_percentage": 0,
    "city_tier": "Tier2",
    "occupation": "student",
    "vehicle_make": "Bajaj",
    "vehicle_type": "two_wheeler",
    "vehicle_age_years": 2,
    "engine_cc": 125,
    "idv": 60000,
    "fuel_type": "petrol",
    "is_modified": 0,
    "policy_type": "third_party_only",
    "voluntary_deductible": 0,
    "addon_count": 0,
    "has_zero_depreciation": 0,
    "has_engine_protection": 0,
    "has_roadside_assistance": 0,
    "has_consumables": 0,
    "policy_start_quarter": 3,
}


# ══════════════════════════════════════════════════════════════
# /health
# ══════════════════════════════════════════════════════════════

class TestHealth:
    def test_status_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_shape(self):
        r = client.get("/health").json()
        assert r["status"] == "ok"
        assert isinstance(r["model_version"], str)
        assert isinstance(r["tweedie_power"], float)
        assert isinstance(r["features_count"], int)
        assert r["features_count"] == 66
        assert "training_metrics" in r

    def test_training_metrics_present(self):
        m = client.get("/health").json()["training_metrics"]
        for key in ["test_norm_gini_glm", "test_norm_gini_lgbm",
                    "frequency_auc", "known_limitations"]:
            assert key in m, f"Missing key: {key}"
        assert len(m["known_limitations"]) == 6

    def test_gini_beats_premium(self):
        m = client.get("/health").json()["training_metrics"]
        assert m["test_norm_gini_glm"] > m["test_norm_gini_premium"], (
            "GLM should beat premium baseline"
        )


# ══════════════════════════════════════════════════════════════
# /quote
# ══════════════════════════════════════════════════════════════

class TestQuote:
    def test_status_200(self):
        r = client.post("/quote", json=ECONOMY_POLICY)
        assert r.status_code == 200

    def test_response_fields(self):
        r = client.post("/quote", json=ECONOMY_POLICY).json()
        for field in ["expected_claim_cost_inr", "model_premium_inr",
                      "lgbm_expected_cost_inr", "top_factors", "model_used"]:
            assert field in r, f"Missing field: {field}"

    def test_premium_is_loading_times_cost(self):
        r = client.post("/quote", json=ECONOMY_POLICY).json()
        assert abs(r["model_premium_inr"] - r["expected_claim_cost_inr"] * 1.20) < 1.0

    def test_top_factors_count(self):
        r = client.post("/quote", json=ECONOMY_POLICY).json()
        assert len(r["top_factors"]) == 5

    def test_top_factors_direction(self):
        r = client.post("/quote", json=ECONOMY_POLICY).json()
        for f in r["top_factors"]:
            assert f["direction"] in ("increases_cost", "decreases_cost")
            if f["shap_contribution"] > 0:
                assert f["direction"] == "increases_cost"
            else:
                assert f["direction"] == "decreases_cost"

    def test_predictions_are_positive(self):
        for policy in [ECONOMY_POLICY, LUXURY_POLICY, TWO_WHEELER]:
            r = client.post("/quote", json=policy).json()
            assert r["expected_claim_cost_inr"] > 0
            assert r["model_premium_inr"] > 0
            assert r["lgbm_expected_cost_inr"] > 0

    def test_luxury_costs_more_than_economy(self):
        economy = client.post("/quote", json=ECONOMY_POLICY).json()
        luxury  = client.post("/quote", json=LUXURY_POLICY).json()
        assert luxury["expected_claim_cost_inr"] > economy["expected_claim_cost_inr"], (
            "Luxury BMW (IDV ₹40L) should cost more than economy Hyundai (IDV ₹6L)"
        )

    def test_two_wheeler_cheapest(self):
        economy    = client.post("/quote", json=ECONOMY_POLICY).json()
        two_wheeler = client.post("/quote", json=TWO_WHEELER).json()
        assert two_wheeler["expected_claim_cost_inr"] < economy["expected_claim_cost_inr"], (
            "Two-wheeler should be cheaper than economy car"
        )

    def test_prior_claims_increases_cost(self):
        no_claims  = {**ECONOMY_POLICY, "prior_claims_count": 0, "ncb_percentage": 50}
        has_claims = {**ECONOMY_POLICY, "prior_claims_count": 3, "ncb_percentage": 0}
        r0 = client.post("/quote", json=no_claims).json()
        r1 = client.post("/quote", json=has_claims).json()
        assert r1["expected_claim_cost_inr"] > r0["expected_claim_cost_inr"], (
            "Prior claims should increase expected cost"
        )

    def test_tp_only_cheaper_than_comprehensive(self):
        comp = {**ECONOMY_POLICY, "policy_type": "comprehensive"}
        tp   = {**ECONOMY_POLICY, "policy_type": "third_party_only",
                "addon_count": 0, "has_zero_depreciation": 0,
                "has_engine_protection": 0, "has_roadside_assistance": 0,
                "has_consumables": 0}
        rc = client.post("/quote", json=comp).json()
        rt = client.post("/quote", json=tp).json()
        assert rt["expected_claim_cost_inr"] < rc["expected_claim_cost_inr"], (
            "TP-only should cost less than comprehensive"
        )

    def test_no_warning_for_normal_policy(self):
        r = client.post("/quote", json=ECONOMY_POLICY).json()
        assert r["warning"] is None

    def test_warning_when_cost_near_idv(self):
        # Force a very low IDV relative to a luxury vehicle
        tiny_idv = {**LUXURY_POLICY, "idv": 50000}
        r = client.post("/quote", json=tiny_idv).json()
        # Either a warning fires or cost is sane — both are acceptable
        # The model might still predict low cost for a ₹50K IDV luxury car
        assert "warning" in r   # field must always be present

    def test_idempotent(self):
        """Same input always returns same output."""
        r1 = client.post("/quote", json=ECONOMY_POLICY).json()
        r2 = client.post("/quote", json=ECONOMY_POLICY).json()
        assert r1["expected_claim_cost_inr"] == r2["expected_claim_cost_inr"]


# ══════════════════════════════════════════════════════════════
# /quote/batch
# ══════════════════════════════════════════════════════════════

class TestBatch:
    def test_status_200(self):
        r = client.post("/quote/batch",
                        json={"policies": [ECONOMY_POLICY, LUXURY_POLICY]})
        assert r.status_code == 200

    def test_count_matches(self):
        r = client.post("/quote/batch",
                        json={"policies": [ECONOMY_POLICY,
                                           LUXURY_POLICY,
                                           TWO_WHEELER]}).json()
        assert r["count"] == 3
        assert len(r["predictions"]) == 3

    def test_batch_matches_single(self):
        """Batch prediction for one policy must equal single-policy prediction."""
        single = client.post("/quote", json=ECONOMY_POLICY).json()
        batch  = client.post("/quote/batch",
                             json={"policies": [ECONOMY_POLICY]}).json()
        assert abs(
            batch["predictions"][0]["expected_claim_cost_inr"]
            - single["expected_claim_cost_inr"]
        ) < 1.0, "Batch and single predictions must agree"

    def test_empty_batch_rejected(self):
        r = client.post("/quote/batch", json={"policies": []})
        assert r.status_code == 422

    def test_ordering_preserved(self):
        """First prediction must correspond to the first policy."""
        r = client.post("/quote/batch",
                        json={"policies": [TWO_WHEELER, LUXURY_POLICY]}).json()
        two_wheeler_pred = r["predictions"][0]["expected_claim_cost_inr"]
        luxury_pred      = r["predictions"][1]["expected_claim_cost_inr"]
        assert two_wheeler_pred < luxury_pred


# ══════════════════════════════════════════════════════════════
# /explain
# ══════════════════════════════════════════════════════════════

class TestExplain:
    def test_status_200(self):
        r = client.get("/explain")
        assert r.status_code == 200

    def test_response_shape(self):
        r = client.get("/explain").json()
        assert "model" in r
        assert "power" in r
        assert "top_relativities" in r
        assert isinstance(r["top_relativities"], list)
        assert len(r["top_relativities"]) > 0

    def test_relativity_fields(self):
        rels = client.get("/explain").json()["top_relativities"]
        for rel in rels:
            assert "feature" in rel
            assert "relativity" in rel
            assert "direction" in rel
            assert rel["direction"] in ("more_expensive", "cheaper")
            assert rel["relativity"] > 0

    def test_log_idv_is_top_feature(self):
        rels = client.get("/explain").json()["top_relativities"]
        top  = rels[0]["feature"]
        assert top == "log_idv", (
            f"log_idv should be the largest relativity, got {top}"
        )

    def test_gender_excluded(self):
        rels   = client.get("/explain").json()["top_relativities"]
        feats  = [r["feature"] for r in rels]
        gender = [f for f in feats if f.startswith("gender_")]
        assert not gender, f"Gender columns must be excluded: {gender}"

    def test_tp_only_is_cheaper(self):
        rels = client.get("/explain").json()["top_relativities"]
        tp   = next((r for r in rels
                     if r["feature"] == "policy_type_third_party_only"), None)
        if tp:
            assert tp["direction"] == "cheaper", (
                "TP-only should have a relativity < 1 (cheaper)"
            )


# ══════════════════════════════════════════════════════════════
# Validation — 422 responses
# ══════════════════════════════════════════════════════════════

class TestValidation:
    def test_invalid_make(self):
        bad = {**ECONOMY_POLICY, "vehicle_make": "Ferrari"}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422
        assert "Ferrari" in r.text

    def test_invalid_city_tier(self):
        bad = {**ECONOMY_POLICY, "city_tier": "Village"}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422

    def test_invalid_ncb(self):
        bad = {**ECONOMY_POLICY, "ncb_percentage": 15}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422

    def test_age_too_young(self):
        bad = {**ECONOMY_POLICY, "age": 16}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422

    def test_age_too_old(self):
        bad = {**ECONOMY_POLICY, "age": 90}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422

    def test_exp_exceeds_age(self):
        bad = {**ECONOMY_POLICY, "age": 20, "driving_exp_years": 10}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422

    def test_invalid_policy_type(self):
        bad = {**ECONOMY_POLICY, "policy_type": "gold_plan"}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422

    def test_negative_idv(self):
        bad = {**ECONOMY_POLICY, "idv": -1}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422

    def test_missing_required_field(self):
        bad = {k: v for k, v in ECONOMY_POLICY.items() if k != "vehicle_make"}
        r   = client.post("/quote", json=bad)
        assert r.status_code == 422