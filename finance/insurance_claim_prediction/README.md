# 🚗 Indian Motor Insurance Claim Cost Prediction

End-to-end insurance analytics project predicting expected claim cost
per policy — simulating actuarial pricing for an Indian motor insurer
under IRDAI regulation, with a production FastAPI serving layer.

---

## 🎯 Problem Statement

Predict `claim_amount` (₹) — the expected annual claim cost per policy.

**Type:** Tweedie Regression (zero-inflated + right-skewed target)
**Target:** `claim_amount` in ₹ (0 for 63.5% of policies, >0 for 36.5%)
**Business Goal:** Price motor insurance policies more accurately than
the current premium, reduce cross-subsidies across segments, and
quantify the expected loss for each customer.

---

## 📊 Dataset Overview

Simulated for **Indian motor insurance, 2022-2023**, with
**100,000 policies** across Metro, Tier1, Tier2, Tier3 cities.

| Table | Rows | Description |
|---|---|---|
| policyholders | 100,500 | Driver demographics, NCB, driving history |
| vehicles | 100,000 | Make, age, IDV, engine CC, fuel type |
| policies | 100,000 | Policy type, premium components, add-ons |
| claims | 36,478 | Actual claim records (non-zero only) |
| claim_labels | 100,000 | Target variable (all policies including zeros) |

**After cleaning:** 97,978 policies (2,022 excluded — corrupt labels)

**Target distribution (zero-inflated):**
| Group | Policies | % | Mean Cost |
|---|---|---|---|
| Zero claim (no accident) | 63,522 | 63.5% | ₹0 |
| Non-zero claim | 36,478 | 36.5% | ₹228,619 |
| **All policies** | **100,000** | **100%** | **₹83,396** |

---

## 🆕 What Makes This Project Different

| Concept | First Appearance |
|---|---|
| Zero-inflated target distribution | This project |
| Tweedie regression (compound Poisson-Gamma) | This project |
| Frequency × Severity decomposition | This project |
| Two-part model (classifier × regressor) | This project |
| Normalised Gini as primary metric | This project |
| NCB (No Claim Bonus) — Indian behavioral signal | This project |
| IDV (Insured Declared Value) — contractual payout cap | This project |
| Loss ratio and combined ratio | This project |
| GLM rating relativities (IRDAI-style) | This project |
| Label integrity audit (corrupt targets from missing IDV) | This project |
| FastAPI serving layer with parity tests | This project |
| Training-serving skew test (200 rows) | This project |

---

## 🏗️ Project Workflow

| Notebook | Purpose | Key Concept |
|---|---|---|
| `01_database_setup.ipynb` | Schema, zero-inflated distribution, loss ratio | Tweedie motivation |
| `02_data_quality.ipynb` | Label audit, IDV cap, premium reconstruction | Corrupt-label exclusion |
| `03_eda.ipynb` | Frequency vs severity, Tweedie power estimate | p=1.67 confirmed |
| `04_feature_engineering.ipynb` | 66 quote-time features, leakage guard | Premium held out as baseline |
| `05_model_training.ipynb` | CV power selection, GLM, LightGBM, two-part | Normalised Gini |
| `06_model_evaluation.ipynb` | Gain attribution, calibration, SHAP, pricing sim | Adverse selection |
| `api/` | FastAPI serving, parity tests, route tests | Training-serving skew |

---

## 🔑 Key EDA Findings

### The Zero-Inflated Distribution Demands Tweedie

Standard regression assumes a unimodal continuous target.
Motor insurance claim amounts have:
- A discrete mass at exactly 0 (63.5% of policies)
- A heavily right-skewed continuous distribution for positives
- Skewness of 6.08 on non-zero claims
- Top 1% of claims = 23% of total claim cost

This is the textbook case for a Tweedie (compound Poisson-Gamma)
distribution with power p between 1 and 2.

**Estimated Tweedie power:** p = 1.67 (95% CI: 1.62 to 1.72)
CV cannot separate powers in the 1.2-1.9 range — all produce
identical Gini. Power affects calibration (balance), not ranking.

### NCB Is the Strongest Behavioral Signal
| NCB % | Claim Rate | Interpretation |
|---|---|---|
| 0% (no history) | 43.54% | No claim-free record |
| 20% | 30.46% | 1 claim-free year |
| 35% | 26.74% | 3 claim-free years |
| 50% | **23.56%** | 5+ claim-free years |

**1.85x difference** between NCB=0 and NCB=50. IRDAI standard slabs
(0/20/25/35/45/50%) encode behavioral risk precisely.

### Severity Scales With Vehicle Value
| Segment | Mean Severity | Mean Claim/IDV |
|---|---|---|
| luxury | ₹11,98,834 | 0.241 |
| mid | ₹1,57,990 | 0.086 |
| economy | ₹69,158 | 0.070 |
| two_wheeler | ₹6,375 | 0.042 |

**188x difference** between luxury and two-wheeler in absolute terms.
Claim/IDV removes the price-level effect — luxury vehicles are also
relatively more expensive to repair per unit of value.

### Seasonality Is Severity-Only
Monsoon (Jul-Sep) share of claims: 25.2% (flat = 25.0%).
Monsoon median claim/IDV: 1.46x the rest.
Frequency is flat; severity spikes. A quote-time model cannot use
month-of-loss (unknown at quote time) as a feature.

---

## 🛠️ Feature Engineering

**66 features, all quote-time only.** Nothing that exists only after
an accident. Premium columns are held out as the baseline to beat.

### Key Features
| Category | Features | Notes |
|---|---|---|
| Driver | age, driving_exp_years, log_income, ncb_percentage, prior_claims_count | experience_ratio = exp/age |
| Vehicle | vehicle_age_years, engine_cc, log_idv, vehicle_make, vehicle_segment | log_idv is the dominant feature |
| Policy | policy_type, addon_count, has_zero_depreciation, voluntary_deductible | has_od_cover derived from policy_type |
| Engineered | log_idv, experience_ratio, idv_to_annual_income, is_young_driver | is_old_vehicle (age >= 8) |

### Leakage Map
```
POST-CLAIM (excluded):  claim_type, claim_month, claim_date,
                        settlement_days, surveyor_assessment,
                        is_fraud_suspected, monsoon_claim,
                        was_capped_at_idv

GENERATOR ORACLE:       claim_prob  → oracle_claim_prob.csv
                        (evaluation only, never a feature)

PREMIUM (baseline):     total_premium, base_od_premium,
                        tp_premium, addon_premium
                        (the baseline to beat, not a feature)

ARTIFACT FLAGS:         vehicle_age_missing, ncb_missing
                        (encode generator bugs, not real signal)
```

### Cleaning Decisions
**Label corruption fix:** 2,022 policies had null IDV. In the generator,
`None or 500000` evaluates to `None` in Python, so severity computed
as `max(500, None)` = ₹500. All 729 claims on those policies showed
median ₹492 and 100% below ₹2,000 — confirmed corrupt. Excluded.

**IDV payout cap:** 1,273 non-third-party claims exceeded their policy's
IDV (4.76%). Capped at IDV — a contractual rule, not a statistical guess.

**Premium reconstruction:** `total_premium` was null for 1,428 policies.
Reconstructed exactly from components (max diff ₹0.0000 vs known values).

---

## 🤖 Model Results

### Scoreboard (test set, 19,596 policies)

| Model | Norm Gini | Deviance | Balance | Capture@20% |
|---|---|---|---|---|
| Premium (scaled) | 0.6913 | 19.166 | 0.942 | 69.6% |
| IDV only (scaled) | 0.6727 | 19.411 | 0.940 | 69.4% |
| Tweedie GLM | **0.7443** | **17.291** | **0.973** | 73.2% |
| Tweedie LightGBM | 0.7428 | 17.361 | 0.893 | 72.8% |
| Two-part LightGBM | 0.7449 | 17.299 | 0.899 | 73.0% |

**All three ML models are tied** within bootstrap noise. The GLM has
the best balance (0.973) and interpretable coefficients. It is the
primary model in the API.

### Why the Models Beat the Premium — Honest Attribution

| Source of gain | Gini points | % of total |
|---|---|---|
| Total gain over premium | +0.0515 | 100% |
| IDV + policy-type fix | +0.0280 | **54%** |
| All other features | +0.0235 | 46% |

More than half the gain comes from the model learning that third-party-only
policies have much lower expected cost than the generator charges them.
This is a pricing correction a single underwriting rule could replicate,
not a novel ML discovery.

### Frequency and Severity Components
| Component | Metric | Value |
|---|---|---|
| Frequency AUC | has_claim classifier | 0.6755 |
| Oracle AUC ceiling | true P(claim) | 0.6786 |
| Severity Spearman | predicted vs actual | 0.4528 |
| Severity deviance explained | vs constant | 17.4% |

**Frequency is solved** — the model is 0.003 AUC below the oracle.
**Severity is the hard part** — claim type (theft vs own damage) is
unknown at quote time, creating irreducible noise.

---

## 📊 SHAP Feature Importance

| Rank | Feature | Mean SHAP | Share |
|---|---|---|---|
| 1 | log_idv | 0.5424 | 22.0% |
| 2 | has_od_cover | 0.5032 | 20.4% |
| 3 | policy_type_third_party_only | 0.1514 | 6.1% |
| 4 | prior_claims_count | 0.1482 | 6.0% |
| 5 | engine_cc | 0.1148 | 4.7% |
| 6 | vehicle_segment_luxury | 0.1068 | 4.3% |
| 7 | ncb_percentage | 0.0981 | 4.0% |

log_idv and has_od_cover together carry 42% of the model's explanation.
The model is primarily an IDV and coverage ranking with risk adjustments.

---

## 🏷️ GLM Rating Relativities (Top 10)

| Feature | Relativity | Meaning |
|---|---|---|
| log_idv | x2.254 | IDV doubles cost per SD increase |
| has_od_cover | x1.293 | Own-damage cover adds 29% |
| policy_type_comprehensive | x1.224 | vs base |
| prior_claims_count | x1.169 | Each prior claim adds 17% |
| vehicle_type_taxi | x1.143 | Taxis cost 14% more |
| is_young_driver | x1.119 | Under-25 adds 12% |
| ncb_percentage | x0.893 | Each NCB level reduces cost |
| vehicle_type_two_wheeler | x0.885 | Two-wheelers cost 12% less |
| policy_type_third_party_only | x0.773 | TP-only costs 23% less |

These are multiplicative on expected claim cost — exactly the format
IRDAI requires insurers to document and file.

---

## 💰 Business Insights

### Segment Pricing (Current vs Model)
| Segment | Current Ratio | Model Ratio | Verdict |
|---|---|---|---|
| economy | 1.769x overcharged | 0.999x | Premium 77% too high |
| luxury | 0.597x undercharged | 0.880x | Premium 33% too low |
| two_wheeler | 3.214x overcharged | 1.074x | Premium 221% too high |
| TP-only | 2.659x overcharged | 0.822x | Premium 166% too high |

### Adverse Selection Risk
If the 20% most overcharged customers leave (attracted by competitors
using better models), the remaining portfolio loss ratio rises from
106.1% to 118.9% — a 12.8 percentage point deterioration. Over-pricing
good risks is self-defeating.

---

## 🚀 FastAPI Serving Layer

```
api/
├── main.py          # FastAPI app (4 endpoints)
├── schema.py        # Pydantic validation (9 validation tests)
├── model.py         # GLM primary, LightGBM secondary, SHAP
├── features.py      # Pure-Python feature builder
├── Dockerfile
└── tests/
    ├── test_parity.py   # 200/200 parity tests ✅
    └── test_routes.py   # 37/37 route tests ✅
```

### Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model version, metrics, 6 known limitations |
| `/quote` | POST | Single-policy prediction + top 5 SHAP factors |
| `/quote/batch` | POST | Batch prediction (1-1000 policies) |
| `/explain` | GET | GLM rating relativities (IRDAI-style table) |

### Sample Response (`/quote`)
```json
{
  "expected_claim_cost_inr": 50337.34,
  "model_premium_inr": 60404.81,
  "lgbm_expected_cost_inr": 37494.03,
  "top_factors": [
    {"feature": "has_od_cover", "shap_contribution": 0.374,
     "direction": "increases_cost"},
    {"feature": "prior_claims_count", "shap_contribution": -0.148,
     "direction": "decreases_cost"}
  ],
  "model_used": "Tweedie GLM (log link, L2 penalty)",
  "warning": null
}
```

### Parity Test
200 random rows from `modeling_data.csv` are fed through
`features.py` and compared element-by-element against the
training matrix. All 200 must match within tolerance 1e-4.
This catches training-serving skew before deployment.

```
pytest tests/test_parity.py   # 200/200 passed in 4.27s
pytest tests/test_routes.py   # 37/37 passed in 20.93s
```

### Run Locally
```bash
cd api/
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

## ⚠️ Known Limitations (All Confirmed by Data)

| # | Limitation | Evidence |
|---|---|---|
| 1 | Loss ratio ~407% (vs real 70-85%) | NB01 Cell 5 |
| 2 | Monsoon affects severity only, not frequency | NB03 Cell 5: Jul-Sep share 25.2% (flat) |
| 3 | Third-party severity multiplier unused | NB03 Cell 5: TP and OD have same claim/IDV |
| 4 | 74.9% of TP-only claims are own-damage type | NB03 Cell 5: real insurance: 0% |
| 5 | TP-only policies charged own-damage premium | NB03 Cell 7: rel LR 0.33 |
| 6 | Missing vehicle age inflates IDV 1.55x | NB05 Cell 2: confirmed |

**What this means for the results:** Limitations 4 and 5 together explain
54% of the gain over the premium. The remaining 46% (+0.0235 Gini) holds
on artifact-free rows and represents genuine feature signal.

---

## 🛠️ Tech Stack

**Data Processing:** PySpark 4.2.0, SQLite (JDBC), SparkSQL

**ML:** Python, Pandas, Scikit-learn (TweedieRegressor), LightGBM, SHAP

**API:** FastAPI, Uvicorn, Pydantic v2, httpx

**Tests:** pytest (200 parity + 37 route tests)

**Deployment:** Docker (multi-stage build)

---

## 🚀 How to Run

```bash
# Generate database
$env:INSURANCE_DB_PATH = "path\to\insurance.db"
python src/data_generator.py

# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

# Start API
cd api/
uvicorn main:app --reload --port 8000
```

---

## 📁 Folder Structure

```
insurance_claim_prediction/
├── data/
│   ├── insurance.db         # SQLite database (44.5 MB)
│   └── processed/           # Cleaned CSVs, models, artifacts
├── notebooks/
│   ├── 01_database_setup.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── api/
│   ├── main.py
│   ├── schema.py
│   ├── model.py
│   ├── features.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       ├── test_parity.py
│       └── test_routes.py
├── src/
│   └── data_generator.py
└── README.md
```

---

## 📌 Key Learnings vs Loan Default Project

| Concept | Loan Default | Insurance Claim |
|---|---|---|
| Target type | Binary 0/1 + EL regression | Zero-inflated continuous |
| Statistical model | Logistic + Tweedie (attempted) | Tweedie GLM + two-part |
| Primary metric | KS Statistic, AUC | Normalised Gini |
| Label corruption | Temporal leakage | Missing IDV → ₹500 floor |
| Honest attribution | 3 perfect score investigations | 6 generator limitations |
| Regulatory frame | RBI, Basel III | IRDAI |
| New signal | DPD trajectory | NCB, IDV, claim/IDV ratio |
| Production layer | None | FastAPI + parity + route tests |