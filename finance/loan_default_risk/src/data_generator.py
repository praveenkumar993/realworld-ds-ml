"""
===============================================================
 RealWorld-DS-ML | Finance | Loan Default Risk Scoring
 File    : data_generator.py
 Purpose : Generate realistic Indian banking loan data
           and write directly to SQLite database.
           PySpark reads from this database via JDBC.

 Tables Generated:
   1. applicant_profile  — borrower demographics and financials
   2. loan_applications  — 50,000 loan applications
   3. repayment_history  — monthly payment behavior per loan
   4. default_labels     — target variable (PD, LGD, EAD, EL)

 Target Variable:
   is_defaulted (0/1)  — binary classification target
   expected_loss (INR) — regression target (PD x LGD x EAD)

 Default Definition:
   90+ days past due (3 missed EMIs) = NPA = defaulted
   This is the RBI standard for Indian banking.

 Indian Banking Context:
   - CIBIL score range 300-900
   - Loan types: Home, Personal, Vehicle, Business, Education
   - NPA classification per RBI guidelines
   - Expected Loss = PD x LGD x EAD (Basel III)
   - LTV ratio for secured loans
   - Debt-to-income ratio as key underwriting feature
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

N_APPLICATIONS    = 50_000
APPLICATION_START = date(2021, 1, 1)
APPLICATION_END   = date(2023, 6, 30)
OBSERVATION_DATE  = date(2024, 12, 31)

CITY_TIERS = {
    "Metro": {
        "cities"     : ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata"],
        "weight"     : 0.35,
        "income_mult": 2.5,
    },
    "Tier1": {
        "cities"     : ["Pune","Ahmedabad","Jaipur","Surat","Lucknow","Nagpur"],
        "weight"     : 0.30,
        "income_mult": 1.5,
    },
    "Tier2": {
        "cities"     : ["Bhopal","Patna","Vadodara","Ludhiana","Nashik","Agra"],
        "weight"     : 0.25,
        "income_mult": 1.0,
    },
    "Tier3": {
        "cities"     : ["Mysuru","Jodhpur","Madurai","Raipur","Guwahati","Coimbatore"],
        "weight"     : 0.10,
        "income_mult": 0.7,
    },
}

LOAN_TYPES = {
    "home_loan"    : {"base_default": 0.04, "weight": 0.30,
                      "avg_amount": 2500000, "tenure_range": (120, 300),
                      "rate_range": (8.5, 10.5),  "is_secured": True},
    "personal_loan": {"base_default": 0.12, "weight": 0.25,
                      "avg_amount": 350000,  "tenure_range": (12, 60),
                      "rate_range": (12.0, 18.0), "is_secured": False},
    "vehicle_loan" : {"base_default": 0.06, "weight": 0.20,
                      "avg_amount": 700000,  "tenure_range": (36, 84),
                      "rate_range": (9.5, 12.5),  "is_secured": True},
    "business_loan": {"base_default": 0.15, "weight": 0.15,
                      "avg_amount": 1500000, "tenure_range": (24, 120),
                      "rate_range": (13.0, 18.5), "is_secured": False},
    "education_loan": {"base_default": 0.10, "weight": 0.10,
                       "avg_amount": 800000,  "tenure_range": (60, 120),
                       "rate_range": (9.0, 12.0),  "is_secured": False},
}

EMPLOYMENT_TYPES = {
    "salaried"     : {"weight": 0.55, "income_range": (25000, 500000),
                      "default_adj": -0.02},
    "self_employed": {"weight": 0.25, "income_range": (20000, 300000),
                      "default_adj": +0.03},
    "business"     : {"weight": 0.15, "income_range": (50000, 1000000),
                      "default_adj": +0.04},
    "retired"      : {"weight": 0.05, "income_range": (15000, 100000),
                      "default_adj": -0.01},
}


# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def rdate(s, e):
    """Random date between s and e."""
    return s + timedelta(days=random.randint(0, (e - s).days))


def compute_emi(principal, annual_rate, tenure_months):
    """Standard reducing balance EMI formula."""
    if annual_rate == 0:
        return principal / tenure_months
    r = annual_rate / 12 / 100
    return principal * r * (1 + r) ** tenure_months / (
        (1 + r) ** tenure_months - 1
    )


def compute_default_probability(row):
    """
    Compute probability of default (PD) for a loan application.

    Each risk factor adjusts the base default rate up or down.
    The base rate comes from the loan type (home loans are safer
    than personal loans due to collateral and borrower motivation).

    This mirrors how Indian banks compute internal credit ratings
    for CIBIL score-based underwriting decisions.
    """
    base_pd = LOAN_TYPES[row["loan_type"]]["base_default"]

    # CIBIL score — single most powerful default predictor
    cibil = row["cibil_score"] or 650
    if cibil >= 800:   cibil_adj = -0.08
    elif cibil >= 750: cibil_adj = -0.05
    elif cibil >= 700: cibil_adj = -0.02
    elif cibil >= 650: cibil_adj = +0.02
    elif cibil >= 600: cibil_adj = +0.05
    elif cibil >= 550: cibil_adj = +0.09
    else:              cibil_adj = +0.15

    # Debt-to-income ratio
    dti = row["debt_to_income_ratio"] or 0.35
    if dti < 0.20:   dti_adj = -0.03
    elif dti < 0.35: dti_adj =  0.00
    elif dti < 0.50: dti_adj = +0.04
    elif dti < 0.65: dti_adj = +0.08
    else:            dti_adj = +0.14

    # Employment type
    emp_adj = EMPLOYMENT_TYPES.get(
        row["employment_type"], {}
    ).get("default_adj", 0.0)

    # Employment tenure
    yrs = row["years_employed"] or 1
    if yrs >= 5:   emp_tenure_adj = -0.02
    elif yrs >= 2: emp_tenure_adj =  0.00
    elif yrs >= 1: emp_tenure_adj = +0.02
    else:          emp_tenure_adj = +0.05

    # LTV ratio for secured loans
    ltv = row.get("ltv_ratio") or 0.70
    if row["loan_type"] in ["home_loan", "vehicle_loan"]:
        if ltv < 0.60:   ltv_adj = -0.02
        elif ltv < 0.75: ltv_adj =  0.00
        elif ltv < 0.85: ltv_adj = +0.02
        else:            ltv_adj = +0.05
    else:
        ltv_adj = 0.0

    # Existing loan burden
    n_loans          = row.get("number_of_existing_loans", 0) or 0
    loan_burden_adj  = min(n_loans * 0.01, 0.06)

    # Credit history length
    credit_hist_adj  = (
        -0.01 if (row.get("credit_history_years") or 0) > 3
        else +0.02
    )

    # Co-applicant reduces risk
    co_app_adj = -0.02 if row.get("has_co_applicant", False) else 0.0

    pd_final = (
        base_pd + cibil_adj + dti_adj + emp_adj +
        emp_tenure_adj + ltv_adj + loan_burden_adj +
        credit_hist_adj + co_app_adj
    )
    pd_final += random.uniform(-0.01, 0.01)
    return float(np.clip(pd_final, 0.005, 0.85))


# ================================================================
#  TABLE 1 — APPLICANT PROFILES
# ================================================================

def generate_applicant_profiles():
    print("Generating applicant profiles (50,000)...")
    records = []

    tier_names   = list(CITY_TIERS.keys())
    tier_weights = [CITY_TIERS[t]["weight"] for t in tier_names]
    emp_names    = list(EMPLOYMENT_TYPES.keys())
    emp_weights  = [EMPLOYMENT_TYPES[e]["weight"] for e in emp_names]

    for i in range(N_APPLICATIONS):
        cid  = f"CUST{i+1:06d}"
        tier = random.choices(tier_names, tier_weights)[0]
        city = random.choice(CITY_TIERS[tier]["cities"])
        emp  = random.choices(emp_names, emp_weights)[0]
        mult = CITY_TIERS[tier]["income_mult"]

        age = random.randint(22, 65)
        if random.random() < 0.02:
            age = random.choice([17, 0, 120])

        inc_min, inc_max = EMPLOYMENT_TYPES[emp]["income_range"]
        monthly_income   = round(
            random.uniform(inc_min, inc_max) * mult, 2
        )

        # CIBIL biased toward realistic mid-range distribution
        cibil = random.choices(
            [random.randint(300, 549),
             random.randint(550, 649),
             random.randint(650, 749),
             random.randint(750, 900)],
            weights=[0.10, 0.20, 0.40, 0.30]
        )[0]
        if random.random() < 0.08:
            cibil = None  # no credit history

        yrs_employed  = round(random.uniform(0.5, 30.0), 1)
        n_existing    = random.choices(
            [0, 1, 2, 3, 4, 5],
            weights=[0.25, 0.30, 0.22, 0.13, 0.07, 0.03]
        )[0]
        credit_hist   = round(random.uniform(0, 20), 1)

        monthly_obligations = round(
            monthly_income * random.uniform(0.0, 0.55), 2
        )
        dti = round(monthly_obligations / (monthly_income + 1), 4)

        has_credit_card  = random.random() > 0.35
        property_owned   = random.random() > 0.45
        has_co_applicant = random.random() > 0.60
        gender = random.choices(
            ["Male", "Female", "Other"],
            weights=[0.58, 0.41, 0.01]
        )[0]
        if random.random() < 0.02:
            gender = None

        records.append({
            "customer_id"             : cid,
            "age"                     : age,
            "gender"                  : gender,
            "city"                    : city,
            "city_tier"               : tier,
            "employment_type"         : emp,
            "years_employed"          : yrs_employed,
            "monthly_income"          : monthly_income,
            "monthly_obligations"     : monthly_obligations,
            "debt_to_income_ratio"    : dti,
            "cibil_score"             : cibil,
            "number_of_existing_loans": n_existing,
            "credit_history_years"    : credit_hist,
            "has_credit_card"         : int(has_credit_card),
            "property_owned"          : int(property_owned),
            "has_co_applicant"        : int(has_co_applicant),
            "phone"                   : (
                fake.phone_number()
                if random.random() > 0.02 else None
            ),
        })

    df    = pd.DataFrame(records)
    dupes = df.sample(frac=0.005, random_state=SEED)
    df    = pd.concat([df, dupes], ignore_index=True)
    print(f"  -> {len(df):,} rows | {df.isnull().sum().sum():,} nulls")
    return df


# ================================================================
#  TABLE 2 — LOAN APPLICATIONS
# ================================================================

def generate_loan_applications(profiles_df):
    print("Generating loan applications (50,000)...")
    records = []

    loan_names   = list(LOAN_TYPES.keys())
    loan_weights = [LOAN_TYPES[l]["weight"] for l in loan_names]

    profiles_clean = profiles_df.drop_duplicates(
        "customer_id"
    ).reset_index(drop=True)

    income_cap_multiples = {
        "home_loan": 60, "personal_loan": 20,
        "vehicle_loan": 40, "business_loan": 36,
        "education_loan": 120,
    }

    for idx, prof in enumerate(profiles_clean.itertuples()):
        cid    = prof.customer_id
        income = prof.monthly_income or 30000

        loan_type = random.choices(loan_names, loan_weights)[0]
        lt_info   = LOAN_TYPES[loan_type]

        # Loan amount — income-constrained
        base_amt   = lt_info["avg_amount"]
        income_cap = income * income_cap_multiples[loan_type]
        loan_amt   = round(
            min(base_amt * random.uniform(0.4, 2.0), income_cap), -3
        )
        loan_amt   = max(loan_amt, 50000)

        tenure  = random.randint(*lt_info["tenure_range"])
        rate    = round(random.uniform(*lt_info["rate_range"]), 2)
        emi     = round(compute_emi(loan_amt, rate, tenure), 2)
        ltv     = round(random.uniform(0.50, 0.95), 3) \
                  if lt_info["is_secured"] else None
        proc_fee = round(loan_amt * random.uniform(0.005, 0.025), 2)
        app_date = rdate(APPLICATION_START, APPLICATION_END)

        # Approval probability based on CIBIL and DTI
        cibil = prof.cibil_score or 650
        dti   = prof.debt_to_income_ratio or 0.3
        if cibil >= 750 and dti < 0.40:
            ap = 0.92
        elif cibil >= 700 and dti < 0.50:
            ap = 0.78
        elif cibil >= 650 and dti < 0.55:
            ap = 0.60
        elif cibil >= 600:
            ap = 0.35
        else:
            ap = 0.15

        status = random.choices(
            ["approved", "rejected", "pending"],
            weights=[ap, (1 - ap) * 0.85, (1 - ap) * 0.15]
        )[0]

        sanctioned = (
            round(loan_amt * random.uniform(0.80, 1.0), -3)
            if status == "approved" else None
        )

        records.append({
            "loan_id"                : f"LN{idx+1:08d}",
            "customer_id"            : cid,
            "application_date"       : app_date.isoformat(),
            "loan_type"              : loan_type,
            "loan_amount_requested"  : loan_amt,
            "loan_amount_sanctioned" : sanctioned,
            "tenure_months"          : tenure,
            "interest_rate"          : rate,
            "emi_amount"             : emi,
            "ltv_ratio"              : ltv,
            "processing_fee"         : proc_fee,
            "application_status"     : status,
            "loan_purpose"           : random.choice([
                "property_purchase", "debt_consolidation",
                "home_renovation",   "vehicle_purchase",
                "business_expansion","education",
                "medical", "wedding", "travel", "other",
            ]),
            "is_secured"             : int(lt_info["is_secured"]),
        })

    df = pd.DataFrame(records)
    print(f"  -> {len(df):,} rows | "
          f"approved={( df['application_status']=='approved').sum():,} "
          f"rejected={(df['application_status']=='rejected').sum():,} "
          f"pending={(df['application_status']=='pending').sum():,}")
    return df


# ================================================================
#  TABLE 3 — REPAYMENT HISTORY  (streamed to SQLite)
# ================================================================

def generate_repayment_history(loans_df, profiles_df, conn):
    """
    For each approved loan, generate month-by-month payment records.

    Default definition: 3 consecutive missed payments = 90 DPD = NPA.
    This mirrors RBI's NPA classification rule used by all Indian banks.

    Records are streamed to SQLite in batches to avoid memory issues.
    Returns a DataFrame of default labels for each approved loan.
    """
    print("Generating repayment history (streaming to SQLite)...")

    conn.execute("DROP TABLE IF EXISTS repayment_history")
    conn.execute("""
        CREATE TABLE repayment_history (
            repayment_id            TEXT,
            loan_id                 TEXT,
            customer_id             TEXT,
            month_number            INTEGER,
            due_date                TEXT,
            due_amount              REAL,
            paid_amount             REAL,
            paid_date               TEXT,
            days_past_due           INTEGER,
            payment_status          TEXT,
            outstanding_principal   REAL,
            cumulative_interest_paid REAL
        )
    """)
    conn.commit()

    profile_map = (
        profiles_df.drop_duplicates("customer_id")
                   .set_index("customer_id")
                   .to_dict("index")
    )

    approved = loans_df[
        loans_df["application_status"] == "approved"
    ].copy().reset_index(drop=True)

    print(f"  Approved loans to process: {len(approved):,}")

    BATCH_SIZE  = 20_000
    batch       = []
    repay_num   = 0
    total_rows  = 0
    labels      = []

    for _, loan in approved.iterrows():
        loan_id   = loan["loan_id"]
        cid       = loan["customer_id"]
        principal = loan["loan_amount_sanctioned"] or loan["loan_amount_requested"]
        rate      = loan["interest_rate"]
        tenure    = int(loan["tenure_months"])
        emi       = loan["emi_amount"]
        app_date  = date.fromisoformat(loan["application_date"])
        disbursal = app_date + timedelta(days=random.randint(30, 60))

        prof = profile_map.get(cid, {})

        # Build risk row for PD computation
        risk_row = {
            "loan_type"               : loan["loan_type"],
            "cibil_score"             : prof.get("cibil_score", 650),
            "debt_to_income_ratio"    : prof.get("debt_to_income_ratio", 0.35),
            "employment_type"         : prof.get("employment_type", "salaried"),
            "years_employed"          : prof.get("years_employed", 3),
            "ltv_ratio"               : loan["ltv_ratio"],
            "number_of_existing_loans": prof.get("number_of_existing_loans", 1),
            "credit_history_years"    : prof.get("credit_history_years", 3),
            "has_co_applicant"        : bool(prof.get("has_co_applicant", 0)),
        }
        pd_score    = compute_default_probability(risk_row)
        will_default = random.random() < pd_score

        # Determine which month default starts (3 missed EMIs = default)
        max_month = min(
            tenure,
            max(3, int((OBSERVATION_DATE - disbursal).days / 30))
        )

        if will_default:
            phase_ranges = [("early", 0.15, 3, min(6, max_month))]
            if max_month >= 7:
                phase_ranges.append(("mid", 0.60, 7, min(24, max_month)))
            if max_month >= 25:
                phase_ranges.append(("late", 0.25, 25, max_month))

            phases, weights, lower_bounds, upper_bounds = zip(*phase_ranges)
            phase = random.choices(phases, weights=weights)[0]
            phase_index = phases.index(phase)
            dm = random.randint(
                lower_bounds[phase_index], upper_bounds[phase_index]
            )
        else:
            dm = None

        outstanding         = float(principal)
        cumulative_interest = 0.0
        monthly_rate        = rate / 12 / 100

        is_defaulted         = False
        default_month_actual = None
        ead                  = 0.0

        for m in range(1, max_month + 1):
            due_date            = disbursal + timedelta(days=30 * m)
            interest_comp       = round(outstanding * monthly_rate, 2)
            principal_comp      = round(
                min(emi - interest_comp, outstanding), 2
            )
            due_amount          = round(emi, 2)

            # Payment behavior logic
            if will_default and dm is not None and m >= dm:
                paid_amount   = 0.0
                days_past_due = (m - dm + 1) * 30
                paid_date_str = None

                if days_past_due >= 90 and not is_defaulted:
                    status        = "defaulted"
                    is_defaulted  = True
                    default_month_actual = m
                    ead           = round(outstanding, 2)
                elif days_past_due < 90:
                    status = "missed"
                else:
                    status = "post_default"
            else:
                rand = random.random()
                if rand < 0.85:
                    paid_amount   = due_amount
                    days_past_due = 0
                    status        = "on_time"
                    paid_date_str = (
                        due_date + timedelta(days=random.randint(-5, 3))
                    ).isoformat()
                elif rand < 0.93:
                    paid_amount   = due_amount
                    days_past_due = random.randint(1, 15)
                    status        = "late"
                    paid_date_str = (
                        due_date + timedelta(days=days_past_due)
                    ).isoformat()
                elif rand < 0.97:
                    paid_amount   = round(
                        due_amount * random.uniform(0.5, 0.9), 2
                    )
                    days_past_due = random.randint(5, 25)
                    status        = "partial"
                    paid_date_str = (
                        due_date + timedelta(days=days_past_due)
                    ).isoformat()
                else:
                    paid_amount   = 0.0
                    days_past_due = 30
                    status        = "missed"
                    paid_date_str = None

            cumulative_interest += interest_comp
            outstanding          = max(0.0, outstanding - principal_comp)

            repay_num += 1
            batch.append((
                f"RP{repay_num:010d}",
                loan_id, cid, m,
                due_date.isoformat(),
                due_amount,
                round(paid_amount, 2),
                paid_date_str,
                days_past_due,
                status,
                round(outstanding, 2),
                round(cumulative_interest, 2),
            ))

            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT INTO repayment_history VALUES"
                    "(?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch
                )
                conn.commit()
                total_rows += len(batch)
                batch.clear()

            if status == "post_default":
                break

        # Compute LGD and recovery
        if is_defaulted:
            is_sec = bool(loan.get("is_secured", 0))
            rec_rate = (
                random.uniform(0.50, 0.85) if is_sec
                else random.uniform(0.10, 0.45)
            )
            recovery_amount = round(ead * rec_rate, 2)
            lgd             = round(1 - rec_rate, 4)
        else:
            recovery_amount = 0.0
            lgd             = 0.0

        loan_face = float(
            loan["loan_amount_sanctioned"] or principal
        )
        expected_loss = round(pd_score * lgd * loan_face, 2)

        labels.append({
            "loan_id"            : loan_id,
            "customer_id"        : cid,
            "is_defaulted"       : int(is_defaulted),
            "default_month"      : default_month_actual,
            "pd_score"           : round(pd_score, 6),
            "ead"                : round(ead, 2),
            "recovery_amount"    : round(recovery_amount, 2),
            "lgd"                : round(lgd, 4),
            "expected_loss"      : round(expected_loss, 2),
            "loan_amount"        : round(loan_face, 2),
            "loan_type"          : loan["loan_type"],
        })

    # Flush remaining batch
    if batch:
        conn.executemany(
            "INSERT INTO repayment_history VALUES"
            "(?,?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
        conn.commit()
        total_rows += len(batch)

    row_count = conn.execute(
        "SELECT COUNT(*) FROM repayment_history"
    ).fetchone()[0]
    print(f"  -> {row_count:,} repayment rows written to SQLite")
    return pd.DataFrame(labels)


# ================================================================
#  MAIN
# ================================================================

def main():
    DB_PATH = os.environ.get("LOAN_DB_PATH") or os.path.join(
        os.path.dirname(__file__), "..", "data", "loan_default.db"
    )
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

    print("=" * 65)
    print(" RealWorld-DS-ML | Finance | Loan Default Risk Scoring")
    print(" Country        : India")
    print(" Context        : Retail banking (RBI NPA guidelines)")
    print(" Applications   : Jan 2021 - Jun 2023")
    print(" Observation    : Until Dec 2024")
    print(" Problem type   : Binary Classification + EL Regression")
    print(" Target 1       : is_defaulted (0/1)")
    print(" Target 2       : expected_loss (INR)")
    print(" Default def    : 90+ DPD = NPA per RBI")
    print("=" * 65)
    print(f" Applications : {N_APPLICATIONS:,}")
    print(f" Database     : {os.path.abspath(DB_PATH)}")
    print("=" * 65)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("  Removed old database")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=100000")

    try:
        profiles_df = generate_applicant_profiles()
        loans_df    = generate_loan_applications(profiles_df)

        print()
        labels_df   = generate_repayment_history(
            loans_df, profiles_df, conn
        )

        print("\nWriting tables to SQLite...")
        profiles_df.to_sql(
            "applicant_profile", conn,
            if_exists="replace", index=False, chunksize=5000
        )
        print(f"  applicant_profile  : {len(profiles_df):,} rows ✅")

        loans_df.to_sql(
            "loan_applications", conn,
            if_exists="replace", index=False, chunksize=5000
        )
        print(f"  loan_applications  : {len(loans_df):,} rows ✅")

        labels_df.to_sql(
            "default_labels", conn,
            if_exists="replace", index=False, chunksize=5000
        )
        print(f"  default_labels     : {len(labels_df):,} rows ✅")

        print("\nCreating indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_prof_cid "
            "ON applicant_profile(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_loan_cid "
            "ON loan_applications(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_loan_id  "
            "ON loan_applications(loan_id)",
            "CREATE INDEX IF NOT EXISTS idx_rep_lid  "
            "ON repayment_history(loan_id)",
            "CREATE INDEX IF NOT EXISTS idx_rep_cid  "
            "ON repayment_history(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_def_lid  "
            "ON default_labels(loan_id)",
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
    for t in ["applicant_profile", "loan_applications",
              "repayment_history",  "default_labels"]:
        n = conn2.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<25}: {n:>10,} rows")

    print(f"\n Default Statistics by Loan Type:")
    rows = conn2.execute("""
        SELECT
            loan_type,
            COUNT(*)                          AS total,
            SUM(is_defaulted)                 AS defaults,
            ROUND(AVG(is_defaulted)*100, 2)   AS default_rate_pct,
            ROUND(AVG(expected_loss), 2)       AS avg_expected_loss
        FROM default_labels
        GROUP BY loan_type
        ORDER BY default_rate_pct DESC
    """).fetchall()

    print(f"  {'Loan Type':<18} {'Total':>8} {'Defaults':>10} "
          f"{'Default%':>10} {'Avg EL (INR)':>14}")
    print(f"  {'-'*64}")
    for r in rows:
        print(f"  {r[0]:<18} {r[1]:>8,} {r[2]:>10,} "
              f"{r[3]:>9.1f}% INR {r[4]:>10,.0f}")

    overall = conn2.execute("""
        SELECT COUNT(*), SUM(is_defaulted),
               ROUND(AVG(is_defaulted)*100, 2),
               ROUND(SUM(expected_loss), 0)
        FROM default_labels
    """).fetchone()
    print(f"\n  Overall: {overall[1]:,} defaults / {overall[0]:,} loans "
          f"({overall[2]}% default rate)")
    print(f"  Total Portfolio Expected Loss: INR {overall[3]:,.0f}")
    conn2.close()

    print(f"\n{'='*65}")
    print(" Next step -> notebooks/01_database_setup.ipynb")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()