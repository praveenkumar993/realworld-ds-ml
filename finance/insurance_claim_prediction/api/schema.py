"""
schema.py — Pydantic request and response models for the insurance API.

Every field that enters the feature builder is validated here.
Unknown vehicle makes and fuel types are rejected at the door so the
one-hot encoder never sees a category it wasn't trained on.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ── Valid category sets (must match feature_spec.json) ────────
VALID_MAKES = {
    "Audi", "BMW", "Bajaj", "Honda", "Hyundai", "Kia",
    "MG", "Mahindra", "Maruti Suzuki", "Mercedes-Benz", "Tata", "Toyota",
}
VALID_SEGMENTS = {"economy", "luxury", "mid", "premium", "two_wheeler"}
VALID_VEHICLE_TYPES = {
    "commercial_vehicle", "private_car", "taxi", "two_wheeler"
}
VALID_FUEL = {"cng", "diesel", "electric", "petrol"}
VALID_POLICY = {
    "comprehensive", "standalone_od", "third_party_only"
}
VALID_TIERS = {"Metro", "Tier1", "Tier2", "Tier3"}
VALID_OCCUPATIONS = {
    "business", "retired", "salaried", "self_employed", "student"
}
NCB_SLABS = {0, 20, 25, 35, 45, 50}

# Make → segment lookup (mirrors the generator)
MAKE_SEGMENT: dict[str, str] = {
    "Maruti Suzuki": "economy",
    "Hyundai":       "economy",
    "Tata":          "economy",
    "Honda":         "mid",
    "Toyota":        "mid",
    "Mahindra":      "mid",
    "Kia":           "mid",
    "MG":            "premium",
    "BMW":           "luxury",
    "Mercedes-Benz": "luxury",
    "Audi":          "luxury",
    "Bajaj":         "two_wheeler",
}


# ── Request ────────────────────────────────────────────────────
class QuoteRequest(BaseModel):
    """
    All fields that a customer would supply when requesting a quote.
    Nothing here exists only after an accident.
    """

    # Policyholder
    age: int = Field(..., ge=18, le=85,
                     description="Driver age in years (18-85)")
    driving_exp_years: float = Field(..., ge=0.5, le=60.0,
                                     description="Years of driving experience")
    monthly_income: float = Field(..., gt=0,
                                  description="Monthly income in INR")
    prior_claims_count: int = Field(0, ge=0, le=10,
                                    description="Number of prior claims")
    ncb_percentage: int = Field(0,
                                description="No Claim Bonus % (0/20/25/35/45/50)")
    city_tier: str = Field(..., description="Metro / Tier1 / Tier2 / Tier3")
    occupation: str = Field(...,
                            description="business/retired/salaried/self_employed/student")

    # Vehicle
    vehicle_make: str = Field(..., description="e.g. Maruti Suzuki, BMW")
    vehicle_type: str = Field(...,
                              description="private_car/two_wheeler/commercial_vehicle/taxi")
    vehicle_age_years: float = Field(..., ge=0, le=30,
                                     description="Vehicle age in years")
    engine_cc: int = Field(..., ge=50, le=6000,
                           description="Engine cubic capacity")
    idv: float = Field(..., gt=0,
                       description="Insured Declared Value in INR")
    fuel_type: str = Field("petrol",
                           description="petrol/diesel/electric/cng")
    is_modified: int = Field(0, ge=0, le=1,
                             description="1 if vehicle has been modified")

    # Policy
    policy_type: str = Field("comprehensive",
                             description="comprehensive/standalone_od/third_party_only")
    voluntary_deductible: float = Field(0.0, ge=0,
                                        description="Voluntary deductible in INR")
    addon_count: int = Field(0, ge=0, le=7,
                             description="Number of add-on covers")
    has_zero_depreciation: int = Field(0, ge=0, le=1)
    has_engine_protection: int = Field(0, ge=0, le=1)
    has_roadside_assistance: int = Field(0, ge=0, le=1)
    has_consumables: int = Field(0, ge=0, le=1)
    policy_start_quarter: int = Field(..., ge=1, le=4,
                                      description="Quarter in which policy starts (1-4)")

    @model_validator(mode="after")
    def validate_categories(self) -> "QuoteRequest":
        if self.vehicle_make not in VALID_MAKES:
            raise ValueError(
                f"vehicle_make '{self.vehicle_make}' not recognised. "
                f"Valid: {sorted(VALID_MAKES)}"
            )
        if self.vehicle_type not in VALID_VEHICLE_TYPES:
            raise ValueError(
                f"vehicle_type must be one of {sorted(VALID_VEHICLE_TYPES)}"
            )
        if self.fuel_type not in VALID_FUEL:
            raise ValueError(
                f"fuel_type must be one of {sorted(VALID_FUEL)}"
            )
        if self.policy_type not in VALID_POLICY:
            raise ValueError(
                f"policy_type must be one of {sorted(VALID_POLICY)}"
            )
        if self.city_tier not in VALID_TIERS:
            raise ValueError(
                f"city_tier must be one of {sorted(VALID_TIERS)}"
            )
        if self.occupation not in VALID_OCCUPATIONS:
            raise ValueError(
                f"occupation must be one of {sorted(VALID_OCCUPATIONS)}"
            )
        if self.ncb_percentage not in NCB_SLABS:
            raise ValueError(
                f"ncb_percentage must be one of {sorted(NCB_SLABS)}"
            )
        if self.driving_exp_years > self.age - 18 + 1:
            raise ValueError(
                "driving_exp_years cannot exceed age - 18 + 1 year"
            )
        return self

    @property
    def vehicle_segment(self) -> str:
        return MAKE_SEGMENT[self.vehicle_make]


class BatchQuoteRequest(BaseModel):
    policies: list[QuoteRequest] = Field(
        ..., min_length=1, max_length=1000,
        description="List of policies (1-1000)"
    )


# ── Response ───────────────────────────────────────────────────
class ShapExplanation(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    direction: Literal["increases_cost", "decreases_cost"]


class QuoteResponse(BaseModel):
    expected_claim_cost_inr: float = Field(
        description="Model's expected annual claim cost (INR)")
    model_premium_inr: float = Field(
        description="Suggested premium = expected cost × 1.20 loading")
    lgbm_expected_cost_inr: float = Field(
        description="LightGBM secondary estimate (INR)")
    top_factors: list[ShapExplanation] = Field(
        description="Top 5 SHAP factors behind this prediction")
    model_used: str = "Tweedie GLM (log link, L2 penalty)"
    warning: Optional[str] = None


class BatchQuoteResponse(BaseModel):
    predictions: list[QuoteResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    tweedie_power: float
    training_metrics: dict
    features_count: int


class ExplainResponse(BaseModel):
    model: str
    power: float
    top_relativities: list[dict]
    note: str