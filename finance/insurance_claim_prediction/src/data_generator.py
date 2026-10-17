"""
===============================================================
 RealWorld-DS-ML | Finance | Insurance Claim Amount Prediction
 File    : data_generator.py
 Purpose : Generate realistic Indian motor insurance data
           and write directly to SQLite database.
           PySpark reads from this database via JDBC.

 Tables Generated:
   1. policyholders   — 100,000 customer profiles
   2. vehicles        — vehicle details per policy
   3. policies        — policy terms and premium history
   4. claims          — claim records (zero-inflated)

 Target Variable:
   claim_amount (INR) — total claim paid
   Zero for ~70% of policies (no claim filed)
   Right-skewed for the remaining 30%

 New Statistical Concepts:
   Zero-inflated distribution — most policies have zero claims
   Tweedie distribution — compound Poisson-Gamma
   Frequency × Severity decomposition
   NCB (No Claim Bonus) — Indian insurance behavioral signal
   IDV (Insured Declared Value) — maximum claim cap
   Loss ratio — claims paid / premiums earned
   Combined ratio — loss ratio + expense ratio

 Indian Insurance Context:
   Regulator     : IRDAI (Insurance Regulatory and
                   Development Authority of India)
   Product       : Motor insurance (largest segment, 40% of market)
   Mandatory     : Third party cover under Motor Vehicles Act
   Key feature   : NCB — premium discount for claim-free years
   Season effect : Monsoon (Aug-Sep) drives highest claims

 Architecture:
   SQLite database → PySpark JDBC → SparkSQL → pandas → ML
===============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

N_POLICIES     = 100_000
POLICY_START   = date(2022, 1, 1)
POLICY_END     = date(2023, 12, 31)
OBSERVATION_DATE = date(2024, 12, 31)

# ── City tiers ────────────────────────────────────────────────
CITY_TIERS = {
    "Metro": {
        "cities"      : ["Mumbai","Delhi","Bangalore",
                         "Chennai","Hyderabad","Kolkata"],
        "weight"      : 0.35,
        "risk_mult"   : 1.4,   # higher traffic density
        "income_mult" : 2.5,
    },
    "Tier1": {
        "cities"      : ["Pune","Ahmedabad","Jaipur",
                         "Surat","Lucknow","Nagpur"],
        "weight"      : 0.30,
        "risk_mult"   : 1.1,
        "income_mult" : 1.5,
    },
    "Tier2": {
        "cities"      : ["Bhopal","Patna","Vadodara",
                         "Ludhiana","Nashik","Agra"],
        "weight"      : 0.25,
        "risk_mult"   : 0.9,
        "income_mult" : 1.0,
    },
    "Tier3": {
        "cities"      : ["Mysuru","Jodhpur","Madurai",
                         "Raipur","Guwahati","Coimbatore"],
        "weight"      : 0.10,
        "risk_mult"   : 0.7,
        "income_mult" : 0.7,
    },
}

# ── Vehicle makes and models ──────────────────────────────────
VEHICLE_MAKES = {
    "Maruti Suzuki" : {
        "weight": 0.28, "avg_idv": 450000,
        "repair_mult": 0.8, "segment": "economy"
    },
    "Hyundai"       : {
        "weight": 0.16, "avg_idv": 600000,
        "repair_mult": 0.9, "segment": "economy"
    },
    "Tata"          : {
        "weight": 0.12, "avg_idv": 700000,
        "repair_mult": 0.85, "segment": "economy"
    },
    "Honda"         : {
        "weight": 0.10, "avg_idv": 800000,
        "repair_mult": 1.0, "segment": "mid"
    },
    "Toyota"        : {
        "weight": 0.08, "avg_idv": 1200000,
        "repair_mult": 1.1, "segment": "mid"
    },
    "Mahindra"      : {
        "weight": 0.08, "avg_idv": 900000,
        "repair_mult": 0.9, "segment": "mid"
    },
    "Kia"           : {
        "weight": 0.06, "avg_idv": 1400000,
        "repair_mult": 1.2, "segment": "mid"
    },
    "MG"            : {
        "weight": 0.04, "avg_idv": 1600000,
        "repair_mult": 1.3, "segment": "premium"
    },
    "BMW"           : {
        "weight": 0.02, "avg_idv": 4000000,
        "repair_mult": 2.5, "segment": "luxury"
    },
    "Mercedes-Benz" : {
        "weight": 0.02, "avg_idv": 5000000,
        "repair_mult": 3.0, "segment": "luxury"
    },
    "Audi"          : {
        "weight": 0.02, "avg_idv": 4500000,
        "repair_mult": 2.8, "segment": "luxury"
    },
    "Bajaj"         : {
        "weight": 0.02, "avg_idv": 80000,
        "repair_mult": 0.5, "segment": "two_wheeler"
    },
}

VEHICLE_TYPES = {
    "private_car"       : {"weight": 0.60, "risk_mult": 1.0},
    "two_wheeler"       : {"weight": 0.25, "risk_mult": 0.6},
    "commercial_vehicle": {"weight": 0.10, "risk_mult": 1.8},
    "taxi"              : {"weight": 0.05, "risk_mult": 2.2},
}

POLICY_TYPES = {
    "comprehensive"   : {"weight": 0.65, "coverage_mult": 1.0},
    "third_party_only": {"weight": 0.30, "coverage_mult": 0.3},
    "standalone_od"   : {"weight": 0.05, "coverage_mult": 0.8},
}

ADDON_COVERS = [
    "zero_depreciation",
    "engine_protection",
    "roadside_assistance",
    "consumables_cover",
    "return_to_invoice",
    "key_replacement",
    "tyre_protection",
]

CLAIM_TYPES = {
    "own_damage"        : {"weight": 0.55, "severity_mult": 1.0},
    "third_party"       : {"weight": 0.25, "severity_mult": 1.5},
    "theft"             : {"weight": 0.08, "severity_mult": 4.0},
    "natural_calamity"  : {"weight": 0.07, "severity_mult": 2.5},
    "fire"              : {"weight": 0.05, "severity_mult": 3.0},
}

# Indian months — monsoon drives highest claims
MONTHLY_RISK = {
    1: 0.85, 2: 0.80, 3: 0.85, 4: 0.90,
    5: 0.95, 6: 1.00, 7: 1.20, 8: 1.45,  # monsoon peak
    9: 1.40, 10: 1.10, 11: 0.95, 12: 0.90,
}


# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def rdate(s, e):
    return s + timedelta(days=random.randint(0, (e - s).days))


def ncb_to_discount(ncb_pct):
    """NCB percentage to premium discount mapping (IRDAI standard)."""
    if ncb_pct == 0:   return 0.00
    elif ncb_pct == 20: return 0.20
    elif ncb_pct == 25: return 0.25
    elif ncb_pct == 35: return 0.35
    elif ncb_pct == 45: return 0.45
    elif ncb_pct == 50: return 0.50
    return 0.0


def compute_claim_probability(row):
    """
    Compute probability that a policy results in a claim.

    Base rate varies by vehicle type and policy type.
    Risk factors adjust the base rate up or down.
    This is the FREQUENCY component of the Tweedie model.
    """
    # Base claim rate by vehicle type
    base_rate = {
        "private_car"       : 0.25,
        "two_wheeler"       : 0.18,
        "commercial_vehicle": 0.40,
        "taxi"              : 0.55,
    }.get(row["vehicle_type"], 0.25)

    # City risk adjustment
    city_risk = {
        "Metro": 0.08, "Tier1": 0.03,
        "Tier2": -0.02, "Tier3": -0.06,
    }.get(row["city_tier"], 0.0)

    # Driver age risk
    age = row["age"] or 35
    if age < 25:        age_risk = +0.12
    elif age < 30:      age_risk = +0.05
    elif age < 50:      age_risk = -0.02
    else:               age_risk = +0.04

    # NCB risk — high NCB means claim-free history
    ncb = row["ncb_percentage"] or 0
    if ncb >= 50:       ncb_risk = -0.10
    elif ncb >= 35:     ncb_risk = -0.06
    elif ncb >= 20:     ncb_risk = -0.03
    else:               ncb_risk = +0.00

    # Vehicle age risk
    veh_age = row["vehicle_age_years"] or 3
    if veh_age <= 1:    veh_risk = -0.03
    elif veh_age <= 3:  veh_risk =  0.00
    elif veh_age <= 7:  veh_risk = +0.04
    else:               veh_risk = +0.09

    # Prior claims risk
    prior = row["prior_claims_count"] or 0
    prior_risk = min(prior * 0.08, 0.25)

    # Policy type — TP only rarely claims own damage
    policy_risk = (
        -0.05 if row["policy_type"] == "third_party_only" else 0.0
    )

    prob = (base_rate + city_risk + age_risk + ncb_risk +
            veh_risk + prior_risk + policy_risk)
    prob += random.uniform(-0.02, 0.02)
    return float(np.clip(prob, 0.03, 0.85))


def compute_claim_severity(row, claim_prob):
    """
    Compute claim amount given that a claim occurs.

    The severity (how much the claim costs) depends on:
    - IDV (maximum payable)
    - Vehicle make (repair cost)
    - Claim type (own damage vs theft vs calamity)
    - Policy add-ons (zero depreciation changes payout)

    This is the SEVERITY component of the Tweedie model.
    """
    idv = row["idv"] or 500000

    # Claim type determines severity as % of IDV
    claim_type = random.choices(
        list(CLAIM_TYPES.keys()),
        weights=[v["weight"] for v in CLAIM_TYPES.values()]
    )[0]

    sev_mult = CLAIM_TYPES[claim_type]["severity_mult"]

    # Repair cost as % of IDV — lognormal distribution
    # Most claims are minor (5-15% of IDV)
    # Some are total loss (80-100% of IDV)
    if claim_type == "theft":
        # Theft = near total loss of IDV
        base_pct = random.uniform(0.70, 1.00)
    elif claim_type == "fire":
        base_pct = random.uniform(0.40, 0.90)
    elif claim_type == "natural_calamity":
        base_pct = random.uniform(0.20, 0.80)
    else:
        # Own damage — follows lognormal (most small, few large)
        base_pct = min(
            np.random.lognormal(
                mean=-2.5, sigma=1.2
            ), 1.0
        )

    # Vehicle make repair cost modifier
    make_info = VEHICLE_MAKES.get(row["vehicle_make"], {})
    repair_mult = make_info.get("repair_mult", 1.0)

    # Zero depreciation add-on — increases payout
    # (without it, insurer deducts depreciation from parts cost)
    has_zero_dep = row.get("has_zero_depreciation", 0)
    zdep_mult = 1.25 if has_zero_dep else 1.0

    # Policy type adjustment
    if row["policy_type"] == "third_party_only":
        # TP only cannot claim own damage
        if claim_type not in ["third_party"]:
            base_pct *= 0.1  # dramatically reduced

    raw_claim = idv * base_pct * repair_mult * zdep_mult
    raw_claim = max(500, min(raw_claim, idv * 1.1))

    # Add multiplicative noise
    noise = np.random.lognormal(mean=0, sigma=0.15)
    final_claim = round(raw_claim * noise, 2)

    return final_claim, claim_type


# ================================================================
#  TABLE 1 — POLICYHOLDERS
# ================================================================

def generate_policyholders():
    print("Generating policyholders (100,000)...")
    records = []

    tier_names   = list(CITY_TIERS.keys())
    tier_weights = [CITY_TIERS[t]["weight"] for t in tier_names]

    for i in range(N_POLICIES):
        pid  = f"PH{i+1:07d}"
        tier = random.choices(tier_names, tier_weights)[0]
        city = random.choice(CITY_TIERS[tier]["cities"])
        mult = CITY_TIERS[tier]["income_mult"]

        age = random.randint(18, 70)
        if random.random() < 0.02:
            age = random.choice([16, 0, 90])

        gender = random.choices(
            ["Male","Female","Other"],
            weights=[0.72, 0.27, 0.01]
        )[0]
        if random.random() < 0.02:
            gender = None

        monthly_income = round(
            random.uniform(15000, 500000) * mult, 2
        )

        # Years of driving experience
        max_exp = max(1, age - 18)
        driving_exp = round(random.uniform(1, max_exp), 1)

        # Prior claims — most people have zero
        prior_claims = random.choices(
            [0, 1, 2, 3, 4],
            weights=[0.58, 0.25, 0.11, 0.04, 0.02]
        )[0]

        # NCB — No Claim Bonus
        # Standard IRDAI NCB slabs: 0, 20, 25, 35, 45, 50%
        # NCB increases with consecutive claim-free years
        # Resets to 0 after any claim
        if prior_claims > 0:
            ncb = 0  # had claims, NCB reset
        else:
            ncb = random.choices(
                [0, 20, 25, 35, 45, 50],
                weights=[0.20, 0.20, 0.20, 0.15, 0.15, 0.10]
            )[0]
        if random.random() < 0.02:
            ncb = None

        occupation = random.choices([
            "salaried", "self_employed", "business",
            "retired", "student"
        ], weights=[0.50, 0.22, 0.18, 0.07, 0.03])[0]

        records.append({
            "policyholder_id"   : pid,
            "age"               : age,
            "gender"            : gender,
            "city"              : city,
            "city_tier"         : tier,
            "monthly_income"    : monthly_income,
            "occupation"        : occupation,
            "driving_exp_years" : driving_exp,
            "prior_claims_count": prior_claims,
            "ncb_percentage"    : ncb,
            "phone"             : fake.phone_number()
                                  if random.random() > 0.02 else None,
            "email"             : fake.email()
                                  if random.random() > 0.05 else None,
        })

    df    = pd.DataFrame(records)
    dupes = df.sample(frac=0.005, random_state=SEED)
    df    = pd.concat([df, dupes], ignore_index=True)
    print(f"  -> {len(df):,} rows | {df.isnull().sum().sum():,} nulls")
    return df


# ================================================================
#  TABLE 2 — VEHICLES
# ================================================================

def generate_vehicles(policyholders_df):
    print("Generating vehicles (100,000)...")
    records = []

    make_names   = list(VEHICLE_MAKES.keys())
    make_weights = [VEHICLE_MAKES[m]["weight"] for m in make_names]
    vtype_names  = list(VEHICLE_TYPES.keys())
    vtype_weights= [VEHICLE_TYPES[v]["weight"] for v in vtype_names]

    ph_clean = policyholders_df.drop_duplicates(
        "policyholder_id"
    )

    for _, ph in ph_clean.iterrows():
        pid  = ph["policyholder_id"]
        tier = ph["city_tier"]

        # Vehicle make — luxury more likely in Metro
        if tier == "Metro":
            adjusted_weights = [
                w * (3.0 if VEHICLE_MAKES[m]["segment"] == "luxury"
                     else 1.0)
                for m, w in zip(make_names, make_weights)
            ]
        else:
            adjusted_weights = make_weights

        make      = random.choices(make_names, adjusted_weights)[0]
        make_info = VEHICLE_MAKES[make]

        # Vehicle type based on make
        if make == "Bajaj":
            vtype = "two_wheeler"
        else:
            vtype = random.choices(vtype_names, vtype_weights)[0]

        # Vehicle age
        veh_age = random.choices(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15],
            weights=[0.08, 0.12, 0.12, 0.11, 0.10, 0.09,
                     0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02]
        )[0]
        if random.random() < 0.02:
            veh_age = None

        # IDV — Insured Declared Value
        # IDV depreciates with vehicle age (IRDAI depreciation table)
        depreciation_rates = {
            0: 0.05, 1: 0.15, 2: 0.20, 3: 0.30,
            4: 0.40, 5: 0.50
        }
        dep_rate = depreciation_rates.get(
            min(veh_age or 0, 5), 0.50
        )
        base_idv = make_info["avg_idv"] * random.uniform(0.7, 1.4)
        idv      = round(base_idv * (1 - dep_rate), -3)
        idv      = max(idv, 10000)
        if random.random() < 0.02:
            idv = None

        # Engine cubic capacity
        if vtype == "two_wheeler":
            cc = random.choice([100, 110, 125, 150, 200, 250])
        elif make_info["segment"] == "luxury":
            cc = random.choice([1998, 2000, 2996, 3000, 4395])
        else:
            cc = random.choice([796, 998, 1197, 1248, 1462,
                                1498, 1598, 1956, 1998, 2199])
        if random.random() < 0.03:
            cc = None

        reg_year = 2024 - (veh_age or 0)

        records.append({
            "vehicle_id"        : f"VH{_+1:07d}",
            "policyholder_id"   : pid,
            "vehicle_make"      : make,
            "vehicle_segment"   : make_info["segment"],
            "vehicle_type"      : vtype,
            "vehicle_age_years" : veh_age,
            "registration_year" : reg_year,
            "engine_cc"         : cc,
            "idv"               : idv,
            "fuel_type"         : random.choices(
                ["petrol","diesel","electric","cng"],
                weights=[0.52, 0.28, 0.12, 0.08]
            )[0],
            "is_modified"       : int(random.random() < 0.04),
        })

    df = pd.DataFrame(records)
    print(f"  -> {len(df):,} rows | {df.isnull().sum().sum():,} nulls")
    return df


# ================================================================
#  TABLE 3 — POLICIES
# ================================================================

def generate_policies(policyholders_df, vehicles_df):
    print("Generating policies (100,000)...")
    records = []

    ph_map  = policyholders_df.drop_duplicates(
        "policyholder_id"
    ).set_index("policyholder_id").to_dict("index")
    veh_map = vehicles_df.set_index(
        "policyholder_id"
    ).to_dict("index")

    ptype_names   = list(POLICY_TYPES.keys())
    ptype_weights = [POLICY_TYPES[p]["weight"] for p in ptype_names]

    ph_clean = policyholders_df.drop_duplicates(
        "policyholder_id"
    ).reset_index(drop=True)

    for idx, ph in ph_clean.iterrows():
        pid  = ph["policyholder_id"]
        ph_d = ph_map.get(pid, {})
        vd   = veh_map.get(pid, {})

        policy_type = random.choices(ptype_names, ptype_weights)[0]
        start_date  = rdate(POLICY_START, POLICY_END)
        end_date    = date(start_date.year + 1,
                           start_date.month, start_date.day)

        # IDV from vehicle
        idv = vd.get("idv", 500000) or 500000

        # Base premium = IDV × rate
        base_rate = {
            "private_car"       : 0.024,
            "two_wheeler"       : 0.018,
            "commercial_vehicle": 0.035,
            "taxi"              : 0.045,
        }.get(vd.get("vehicle_type","private_car"), 0.024)

        base_od_premium = round(idv * base_rate, 2)

        # Third party premium — IRDAI fixed rates
        tp_premium = {
            "private_car"       : 2094,
            "two_wheeler"       : 714,
            "commercial_vehicle": 7897,
            "taxi"              : 7897,
        }.get(vd.get("vehicle_type","private_car"), 2094)

        # Add-on covers
        n_addons = (
            0 if policy_type == "third_party_only" else
            random.choices([0,1,2,3,4], weights=[0.25,0.30,0.25,0.15,0.05])[0]
        )
        addons = random.sample(ADDON_COVERS, n_addons)
        addon_premium = len(addons) * round(
            idv * random.uniform(0.001, 0.003), 2
        )

        # NCB discount
        ncb = ph_d.get("ncb_percentage", 0) or 0
        ncb_discount  = round(
            base_od_premium * ncb_to_discount(ncb), 2
        )

        # Voluntary deductible
        vol_deductible = random.choices(
            [0, 500, 1000, 2000, 5000],
            weights=[0.50, 0.15, 0.15, 0.12, 0.08]
        )[0]
        deductible_discount = round(
            vol_deductible * 0.5, 2
        )

        total_premium = round(
            base_od_premium - ncb_discount - deductible_discount +
            tp_premium + addon_premium, 2
        )
        total_premium = max(total_premium, 500)

        if random.random() < 0.015:
            total_premium = None

        records.append({
            "policy_id"              : f"POL{idx+1:08d}",
            "policyholder_id"        : pid,
            "policy_type"            : policy_type,
            "policy_start_date"      : start_date.isoformat(),
            "policy_end_date"        : end_date.isoformat(),
            "idv"                    : idv,
            "base_od_premium"        : base_od_premium,
            "tp_premium"             : tp_premium,
            "addon_premium"          : addon_premium,
            "ncb_discount"           : ncb_discount,
            "voluntary_deductible"   : vol_deductible,
            "deductible_discount"    : deductible_discount,
            "total_premium"          : total_premium,
            "has_zero_depreciation"  : int("zero_depreciation"
                                           in addons),
            "has_engine_protection"  : int("engine_protection"
                                           in addons),
            "has_roadside_assistance": int("roadside_assistance"
                                           in addons),
            "has_consumables"        : int("consumables_cover"
                                           in addons),
            "addon_count"            : n_addons,
            "insurer"                : random.choices([
                "HDFC ERGO","Bajaj Allianz","ICICI Lombard",
                "New India Assurance","United India","Tata AIG",
            ], weights=[0.18,0.17,0.17,0.16,0.16,0.16])[0],
        })

    df = pd.DataFrame(records)
    print(f"  -> {len(df):,} rows | {df.isnull().sum().sum():,} nulls")
    return df


# ================================================================
#  TABLE 4 — CLAIMS
# ================================================================

def generate_claims(policyholders_df, vehicles_df,
                    policies_df, conn):
    """
    Generate claim records — one row per policy that filed a claim.
    ~30% of policies will have a claim. The rest have claim_amount=0.

    This creates the ZERO-INFLATED target distribution:
    70% of policies: claim_amount = 0 (no claim filed)
    30% of policies: claim_amount > 0 (claim filed and settled)

    Written directly to SQLite — no in-memory accumulation.
    """
    print("Generating claims (streaming to SQLite)...")

    conn.execute("DROP TABLE IF EXISTS claims")
    conn.execute("""
        CREATE TABLE claims (
            claim_id            TEXT,
            policy_id           TEXT,
            policyholder_id     TEXT,
            claim_date          TEXT,
            claim_type          TEXT,
            claim_amount        REAL,
            surveyor_assessment REAL,
            settlement_days     INTEGER,
            is_fraud_suspected  INTEGER,
            monsoon_claim       INTEGER,
            claim_month         INTEGER,
            claim_year          INTEGER
        )
    """)
    conn.commit()

    # Build lookup maps
    ph_map  = policyholders_df.drop_duplicates(
        "policyholder_id"
    ).set_index("policyholder_id").to_dict("index")
    veh_map = vehicles_df.set_index(
        "policyholder_id"
    ).to_dict("index")
    pol_map = policies_df.set_index(
        "policy_id"
    ).to_dict("index")

    # Also build policy → policyholder mapping
    pol_to_ph = dict(zip(
        policies_df["policy_id"],
        policies_df["policyholder_id"]
    ))

    BATCH_SIZE  = 10_000
    batch       = []
    claim_num   = 0
    total_claims = 0

    claim_labels = []   # for summary statistics

    for idx, pol in policies_df.iterrows():
        pol_id = pol["policy_id"]
        pid    = pol["policyholder_id"]

        ph_d   = ph_map.get(pid, {})
        vd     = veh_map.get(pid, {})

        # Build risk row for probability computation
        risk_row = {
            "vehicle_type"      : vd.get("vehicle_type","private_car"),
            "city_tier"         : ph_d.get("city_tier","Tier1"),
            "age"               : ph_d.get("age", 35),
            "ncb_percentage"    : ph_d.get("ncb_percentage", 0),
            "vehicle_age_years" : vd.get("vehicle_age_years", 3),
            "prior_claims_count": ph_d.get("prior_claims_count", 0),
            "policy_type"       : pol.get("policy_type","comprehensive"),
            "vehicle_make"      : vd.get("vehicle_make","Maruti Suzuki"),
            "idv"               : pol.get("idv", 500000),
            "has_zero_depreciation": pol.get("has_zero_depreciation",0),
        }

        claim_prob = compute_claim_probability(risk_row)
        has_claim  = random.random() < claim_prob

        if not has_claim:
            # No claim — still record in labels for modeling
            claim_labels.append({
                "policy_id"     : pol_id,
                "claim_amount"  : 0.0,
                "claim_prob"    : round(claim_prob, 6),
                "has_claim"     : 0,
            })
            continue

        # Determine claim date within policy period
        start = date.fromisoformat(
            pol["policy_start_date"]
        )
        end   = date.fromisoformat(
            pol["policy_end_date"]
        )
        claim_date = rdate(start, min(end, OBSERVATION_DATE))
        claim_month = claim_date.month
        claim_year  = claim_date.year

        # Apply monsoon risk amplification (July-September)
        monthly_multiplier = MONTHLY_RISK.get(claim_month, 1.0)

        # Severity computation
        severity_row = {**risk_row, **{
            "idv"      : pol.get("idv", 500000),
            "policy_type": pol.get("policy_type","comprehensive"),
        }}
        claim_amount, claim_type = compute_claim_severity(
            severity_row, claim_prob
        )

        # Apply monthly risk to severity too
        claim_amount = round(
            claim_amount * monthly_multiplier, 2
        )

        # Surveyor may assess differently (±15%)
        surveyor_assessment = round(
            claim_amount * random.uniform(0.85, 1.15), 2
        )

        # Settlement time in days
        if claim_amount < 50000:
            settlement_days = random.randint(3, 15)
        elif claim_amount < 200000:
            settlement_days = random.randint(15, 45)
        elif claim_amount < 1000000:
            settlement_days = random.randint(30, 90)
        else:
            settlement_days = random.randint(60, 180)

        # Fraud suspicion — large claims from high-risk profiles
        fraud_flag = int(
            claim_amount > 500000 and
            ph_d.get("prior_claims_count", 0) >= 2 and
            random.random() < 0.15
        )

        monsoon_claim = int(claim_month in [7, 8, 9])

        claim_num  += 1
        total_claims += 1

        batch.append((
            f"CLM{claim_num:08d}",
            pol_id, pid,
            claim_date.isoformat(),
            claim_type,
            round(claim_amount, 2),
            surveyor_assessment,
            settlement_days,
            fraud_flag,
            monsoon_claim,
            claim_month,
            claim_year,
        ))

        claim_labels.append({
            "policy_id"   : pol_id,
            "claim_amount": round(claim_amount, 2),
            "claim_prob"  : round(claim_prob, 6),
            "has_claim"   : 1,
        })

        if len(batch) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO claims VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            batch.clear()

    if batch:
        conn.executemany(
            "INSERT INTO claims VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
        conn.commit()

    row_count = conn.execute(
        "SELECT COUNT(*) FROM claims"
    ).fetchone()[0]
    print(f"  -> {row_count:,} claims written to SQLite")
    print(f"  -> {total_claims:,} policies filed claims "
          f"({total_claims/N_POLICIES*100:.1f}%)")

    return pd.DataFrame(claim_labels)


# ================================================================
#  MAIN
# ================================================================

def main():
    DB_PATH = os.environ.get("INSURANCE_DB_PATH") or os.path.join(
        os.path.dirname(__file__), "..", "data", "insurance.db"
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True
    )

    print("=" * 65)
    print(" RealWorld-DS-ML | Finance | Insurance Claim Prediction")
    print(" Country        : India")
    print(" Product        : Motor Insurance")
    print(" Regulator      : IRDAI")
    print(" Problem type   : Tweedie Regression")
    print("                  (Zero-inflated + severity)")
    print(" Target         : claim_amount (INR)")
    print("                  0 for ~70% policies (no claim)")
    print("                  >0 for ~30% (claim filed)")
    print("=" * 65)
    print(f" Policies : {N_POLICIES:,}")
    print(f" Database : {os.path.abspath(DB_PATH)}")
    print("=" * 65)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("  Removed old database")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=100000")

    try:
        ph_df  = generate_policyholders()
        veh_df = generate_vehicles(ph_df)
        pol_df = generate_policies(ph_df, veh_df)
        labels_df = generate_claims(
            ph_df, veh_df, pol_df, conn
        )

        print("\nWriting tables to SQLite...")
        ph_df.to_sql(
            "policyholders", conn,
            if_exists="replace", index=False, chunksize=5000
        )
        print(f"  policyholders : {len(ph_df):,} rows ✅")

        veh_df.to_sql(
            "vehicles", conn,
            if_exists="replace", index=False, chunksize=5000
        )
        print(f"  vehicles      : {len(veh_df):,} rows ✅")

        pol_df.to_sql(
            "policies", conn,
            if_exists="replace", index=False, chunksize=5000
        )
        print(f"  policies      : {len(pol_df):,} rows ✅")

        labels_df.to_sql(
            "claim_labels", conn,
            if_exists="replace", index=False, chunksize=5000
        )
        print(f"  claim_labels  : {len(labels_df):,} rows ✅")

        print("\nCreating indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_ph_id "
            "ON policyholders(policyholder_id)",
            "CREATE INDEX IF NOT EXISTS idx_veh_ph "
            "ON vehicles(policyholder_id)",
            "CREATE INDEX IF NOT EXISTS idx_pol_ph "
            "ON policies(policyholder_id)",
            "CREATE INDEX IF NOT EXISTS idx_pol_id "
            "ON policies(policy_id)",
            "CREATE INDEX IF NOT EXISTS idx_clm_pol "
            "ON claims(policy_id)",
            "CREATE INDEX IF NOT EXISTS idx_clm_ph "
            "ON claims(policyholder_id)",
            "CREATE INDEX IF NOT EXISTS idx_lbl_pol "
            "ON claim_labels(policy_id)",
        ]
        for sql in indexes:
            conn.execute(sql)
            print(f"  OK: {sql.split('ON')[1].strip()}")
        conn.commit()

    finally:
        conn.close()

    # ── Summary ───────────────────────────────────────────────
    size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"\n{'='*65}")
    print(f" Database ready : {DB_PATH}")
    print(f" File size      : {size_mb:.1f} MB")
    print(f"{'='*65}")

    conn2 = sqlite3.connect(DB_PATH)
    for t in ["policyholders","vehicles","policies",
              "claims","claim_labels"]:
        n = conn2.execute(
            f"SELECT COUNT(*) FROM {t}"
        ).fetchone()[0]
        print(f"  {t:<20}: {n:>10,} rows")

    print(f"\nClaim Amount Distribution:")
    stats = conn2.execute("""
        SELECT
            COUNT(*)                           AS total_policies,
            SUM(CASE WHEN claim_amount = 0
                     THEN 1 ELSE 0 END)        AS zero_claims,
            SUM(CASE WHEN claim_amount > 0
                     THEN 1 ELSE 0 END)        AS with_claims,
            ROUND(AVG(claim_amount), 2)         AS mean_all,
            ROUND(AVG(CASE WHEN claim_amount > 0
                          THEN claim_amount END),2) AS mean_given_claim,
            ROUND(MAX(claim_amount), 2)         AS max_claim,
            ROUND(SUM(claim_amount), 2)         AS total_claims_paid
        FROM claim_labels
    """).fetchone()

    print(f"  Total policies       : {stats[0]:>10,}")
    print(f"  Zero-claim policies  : {stats[1]:>10,} "
          f"({stats[1]/stats[0]*100:.1f}%)")
    print(f"  Claim policies       : {stats[2]:>10,} "
          f"({stats[2]/stats[0]*100:.1f}%)")
    print(f"  Mean (all policies)  : INR {stats[3]:>12,.2f}")
    print(f"  Mean (given claim)   : INR {stats[4]:>12,.2f}")
    print(f"  Max single claim     : INR {stats[5]:>12,.2f}")
    print(f"  Total claims paid    : INR {stats[6]:>12,.2f}")

    print(f"\nClaim Amount by Claim Type:")
    conn2.execute("""
        SELECT
            claim_type,
            COUNT(*)                           AS claims,
            ROUND(AVG(claim_amount), 2)        AS avg_amount,
            ROUND(MAX(claim_amount), 2)        AS max_amount
        FROM claims
        GROUP BY claim_type
        ORDER BY avg_amount DESC
    """).fetchall()
    rows = conn2.execute("""
        SELECT claim_type,
               COUNT(*) as claims,
               ROUND(AVG(claim_amount),2) as avg_amt,
               ROUND(MAX(claim_amount),2) as max_amt
        FROM claims GROUP BY claim_type
        ORDER BY avg_amt DESC
    """).fetchall()
    print(f"  {'Type':<22} {'Claims':>8} "
          f"{'Avg (INR)':>14} {'Max (INR)':>14}")
    print(f"  {'-'*62}")
    for r in rows:
        print(f"  {r[0]:<22} {r[1]:>8,} "
              f"INR {r[2]:>10,.0f} INR {r[3]:>10,.0f}")

    print(f"\nLoss Ratio by Insurer:")
    rows2 = conn2.execute("""
        SELECT
            p.insurer,
            COUNT(DISTINCT p.policy_id)        AS policies,
            ROUND(SUM(p.total_premium), 0)     AS total_premium,
            ROUND(SUM(l.claim_amount), 0)      AS total_claims,
            ROUND(SUM(l.claim_amount) /
                  NULLIF(SUM(p.total_premium),0) * 100, 2)
                                               AS loss_ratio_pct
        FROM policies p
        JOIN claim_labels l ON p.policy_id = l.policy_id
        GROUP BY p.insurer
        ORDER BY loss_ratio_pct DESC
    """).fetchall()
    print(f"  {'Insurer':<22} {'Policies':>10} "
          f"{'Loss Ratio':>12}")
    print(f"  {'-'*48}")
    for r in rows2:
        print(f"  {r[0]:<22} {r[1]:>10,} {r[4]:>11.1f}%")

    conn2.close()
    print(f"\n{'='*65}")
    print(" Next step -> notebooks/01_database_setup.ipynb")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()