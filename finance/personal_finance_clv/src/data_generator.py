"""
===============================================================
 RealWorld-DS-ML | Finance | Personal Finance CLV Prediction
 File    : data_generator.py
 Purpose : Generate realistic Indian banking customer data
           and write directly to a SQLite database.
           PySpark reads from this database via JDBC in notebooks.

 Tables Generated (written to SQLite):
   1. customers     — 50,000 customer profiles
   2. transactions  — ~3 million transaction rows (2 years)
   3. products      — product holdings per customer
   4. clv_labels    — actual CLV for the labeling period

 Target Variable:
   clv_next_12months (₹) — total net revenue Ola bank expects
   from this customer over the next 12 months.
   This is a REGRESSION target — predicting a continuous
   monetary value, not a binary outcome.

 Indian Banking Context:
   - City tiers: Metro (Mumbai, Delhi, Bangalore, Chennai,
     Hyderabad, Kolkata) / Tier 1 / Tier 2 / Tier 3
   - Products: Savings, FD, RD, Mutual Fund, Credit Card,
     Home Loan, Personal Loan, Insurance
   - Channels: UPI, NEFT, RTGS, ATM, POS, NetBanking, Branch
   - Regulatory: KYC status, PAN linkage, Aadhaar linkage
   - Revenue: Transaction fees + loan interest +
     investment commission + credit card interest

 Key Design Decisions:
   - 50,000 customers × ~60 transactions/year × 2 years
     ≈ 3 million transaction rows — genuinely PySpark-scale
   - CLV is computed from a revenue formula:
     loan_interest + investment_commission + txn_fees +
     credit_card_interest - operational_costs
   - Right-skewed CLV: most customers ₹2,000-8,000/year,
     HNI customers ₹50,000-500,000/year
   - City tier drives income, spending, and product complexity
   - Intentional messiness: nulls, duplicates, outliers
===============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta, date
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

# ── Config ────────────────────────────────────────────────────
N_CUSTOMERS = 50_000

# History window: Jan 2023 - Dec 2023 (feature computation)
# Label window : Jan 2024 - Dec 2024 (CLV target)
HISTORY_START = date(2023, 1, 1)
HISTORY_END   = date(2023, 12, 31)
LABEL_START   = date(2024, 1, 1)
LABEL_END     = date(2024, 12, 31)

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "finance_clv.db"
)
DB_FALLBACK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data",
    f"finance_clv_{os.getpid()}_{int(time.time())}.db"
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ── Indian city tiers ─────────────────────────────────────────
CITY_TIERS = {
    "Metro": {
        "cities": ["Mumbai", "Delhi", "Bangalore",
                   "Chennai", "Hyderabad", "Kolkata"],
        "weight": 0.35,
        "income_multiplier": 2.5,
        "hni_probability": 0.15,
    },
    "Tier1": {
        "cities": ["Pune", "Ahmedabad", "Jaipur", "Surat",
                   "Lucknow", "Kanpur", "Nagpur", "Indore"],
        "weight": 0.30,
        "income_multiplier": 1.5,
        "hni_probability": 0.06,
    },
    "Tier2": {
        "cities": ["Bhopal", "Visakhapatnam", "Patna",
                   "Vadodara", "Ludhiana", "Agra", "Nashik",
                   "Faridabad", "Meerut", "Rajkot"],
        "weight": 0.25,
        "income_multiplier": 1.0,
        "hni_probability": 0.02,
    },
    "Tier3": {
        "cities": ["Mysuru", "Jodhpur", "Madurai", "Raipur",
                   "Kota", "Guwahati", "Chandigarh", "Coimbatore",
                   "Vijayawada", "Thiruvananthapuram"],
        "weight": 0.10,
        "income_multiplier": 0.7,
        "hni_probability": 0.005,
    },
}

# ── Customer segments ─────────────────────────────────────────
CUSTOMER_SEGMENTS = {
    "student"       : {"weight": 0.12, "income_range": (0, 15000),
                       "age_range": (18, 24)},
    "salaried_entry": {"weight": 0.25, "income_range": (15000, 40000),
                       "age_range": (22, 35)},
    "salaried_mid"  : {"weight": 0.28, "income_range": (40000, 100000),
                       "age_range": (28, 50)},
    "salaried_senior": {"weight": 0.10, "income_range": (100000, 300000),
                        "age_range": (35, 60)},
    "self_employed" : {"weight": 0.15, "income_range": (20000, 200000),
                       "age_range": (25, 60)},
    "hni"           : {"weight": 0.05, "income_range": (300000, 2000000),
                       "age_range": (30, 65)},
    "retired"       : {"weight": 0.05, "income_range": (10000, 80000),
                       "age_range": (58, 80)},
}

# ── Product types and revenue rates ──────────────────────────
PRODUCT_TYPES = {
    "savings_account"  : {"commission_rate": 0.0,  "interest_rate": 0.0},
    "fixed_deposit"    : {"commission_rate": 0.005, "interest_rate": 0.0},
    "recurring_deposit": {"commission_rate": 0.003, "interest_rate": 0.0},
    "mutual_fund"      : {"commission_rate": 0.01,  "interest_rate": 0.0},
    "credit_card"      : {"commission_rate": 0.0,   "interest_rate": 0.36},
    "home_loan"        : {"commission_rate": 0.0,   "interest_rate": 0.085},
    "personal_loan"    : {"commission_rate": 0.0,   "interest_rate": 0.14},
    "insurance"        : {"commission_rate": 0.15,  "interest_rate": 0.0},
}

# ── Transaction types ─────────────────────────────────────────
TXN_TYPES = {
    "UPI"       : {"weight": 0.45, "fee_rate": 0.0,    "avg_amount": 800},
    "POS"       : {"weight": 0.20, "fee_rate": 0.009,  "avg_amount": 1500},
    "ATM"       : {"weight": 0.10, "fee_rate": 20,     "avg_amount": 3000},
    "NEFT"      : {"weight": 0.08, "fee_rate": 2.5,    "avg_amount": 15000},
    "RTGS"      : {"weight": 0.02, "fee_rate": 25,     "avg_amount": 250000},
    "NetBanking": {"weight": 0.08, "fee_rate": 5,      "avg_amount": 8000},
    "EMI_debit" : {"weight": 0.05, "fee_rate": 0.0,    "avg_amount": 12000},
    "Branch"    : {"weight": 0.02, "fee_rate": 50,     "avg_amount": 20000},
}

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "healthcare",
    "education", "entertainment", "travel", "utilities",
    "ecommerce", "clothing", "electronics", "rent",
    "investment", "insurance", "salary_credit", "other"
]

CHANNELS = ["mobile_app", "netbanking", "atm",
            "branch", "phone_banking", "upi_app"]


# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def random_date(start: date, end: date) -> date:
    """Random date between start and end."""
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def create_transactions_table(conn, table_name: str):
    """Create or replace the transaction table for a fresh run."""
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = DELETE")

    for attempt in range(3):
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"""
                CREATE TABLE {table_name} (
                    txn_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    txn_date TEXT,
                    txn_type TEXT,
                    amount REAL,
                    credit_debit TEXT,
                    merchant_category TEXT,
                    channel TEXT,
                    balance_after REAL,
                    city TEXT,
                    is_international INTEGER
                )
            """)
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < 2:
                time.sleep(2)
                continue
            raise


def insert_records_to_sqlite(conn, table_name: str, records: list):
    """Insert a batch of transaction dictionaries into SQLite."""
    if not records:
        return

    columns = list(records[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    sql = f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})"
    values = [tuple(r[c] for c in columns) for r in records]
    conn.executemany(sql, values)
    conn.commit()


def get_city_and_tier():
    """Pick a city and return both city name and tier."""
    tier_name = random.choices(
        list(CITY_TIERS.keys()),
        weights=[v["weight"] for v in CITY_TIERS.values()]
    )[0]
    city = random.choice(CITY_TIERS[tier_name]["cities"])
    return city, tier_name


def get_segment():
    """Pick a customer segment."""
    return random.choices(
        list(CUSTOMER_SEGMENTS.keys()),
        weights=[v["weight"] for v in CUSTOMER_SEGMENTS.values()]
    )[0]


def monthly_txn_count(segment, city_tier):
    """
    How many transactions does this customer make per month?
    Based on segment and city tier.
    """
    base = {
        "student"        : random.randint(5, 15),
        "salaried_entry" : random.randint(10, 25),
        "salaried_mid"   : random.randint(15, 40),
        "salaried_senior": random.randint(20, 50),
        "self_employed"  : random.randint(25, 60),
        "hni"            : random.randint(30, 80),
        "retired"        : random.randint(5, 20),
    }[segment]

    multiplier = {"Metro": 1.3, "Tier1": 1.1,
                  "Tier2": 0.9, "Tier3": 0.7}[city_tier]
    return max(1, int(base * multiplier))


def compute_clv(customer_row, products_for_customer,
                label_transactions):
    """
    Compute actual CLV for a customer over the label period.

    Revenue streams:
    1. Transaction fees from label period transactions
    2. Loan interest (home_loan, personal_loan)
    3. Investment commission (mutual_fund, FD, RD)
    4. Insurance commission
    5. Credit card interest (revolving balance)
    6. Minus: operational cost per customer

    This mirrors how Indian banks compute relationship value.
    """
    clv = 0.0

    # ── 1. Transaction fees ───────────────────────────────────
    for _, txn in label_transactions.iterrows():
        txn_info = TXN_TYPES.get(txn["txn_type"], {})
        fee_rate = txn_info.get("fee_rate", 0)
        amount   = txn.get("amount", 0) or 0

        if fee_rate < 1:  # percentage fee
            fee = amount * fee_rate
        else:             # flat fee
            fee = fee_rate
        clv += fee

    # ── 2. Product revenue ────────────────────────────────────
    for _, prod in products_for_customer.iterrows():
        if not prod.get("is_active", False):
            continue

        ptype   = prod.get("product_type", "")
        value   = prod.get("current_value", 0) or 0
        rates   = PRODUCT_TYPES.get(ptype, {})

        comm_rate     = rates.get("commission_rate", 0)
        interest_rate = rates.get("interest_rate", 0)

        if ptype in ["home_loan", "personal_loan", "credit_card"]:
            # Bank earns interest on outstanding balance
            # Assume 60% utilization for credit card
            if ptype == "credit_card":
                revolving = value * 0.30  # 30% revolve
                clv += revolving * interest_rate
            else:
                clv += value * interest_rate

        elif ptype in ["mutual_fund", "fixed_deposit",
                       "recurring_deposit"]:
            clv += value * comm_rate

        elif ptype == "insurance":
            clv += value * comm_rate  # value = annual premium

    # ── 3. Operational cost (deduct) ─────────────────────────
    # Each customer costs approximately ₹800/year to service
    clv -= 800

    # ── 4. Minimum CLV floor ──────────────────────────────────
    clv = max(clv, 0)

    # ── 5. Add noise ──────────────────────────────────────────
    noise_factor = random.uniform(0.85, 1.15)
    clv = round(clv * noise_factor, 2)

    return clv


# ================================================================
#  TABLE 1 — CUSTOMERS
# ================================================================

def generate_customers():
    print("Generating customers table (50,000 rows)...")
    records = []

    for i in range(N_CUSTOMERS):
        customer_id = f"CUST{str(i+1).zfill(6)}"
        city, tier  = get_city_and_tier()
        segment     = get_segment()

        seg_info    = CUSTOMER_SEGMENTS[segment]
        tier_info   = CITY_TIERS[tier]

        # Age
        age_min, age_max = seg_info["age_range"]
        age = random.randint(age_min, age_max)
        if random.random() < 0.02:
            age = random.choice([-1, 0, 150])

        # Income
        inc_min, inc_max = seg_info["income_range"]
        monthly_income = round(
            random.uniform(inc_min, inc_max) *
            tier_info["income_multiplier"], 2
        )

        # Account open date
        account_open = random_date(
            date(2018, 1, 1), date(2022, 12, 31)
        )
        account_age_days = (date(2023, 1, 1) - account_open).days

        # KYC and compliance
        kyc_complete    = random.random() > 0.08
        pan_linked      = random.random() > 0.05
        aadhaar_linked  = random.random() > 0.10
        if random.random() < 0.03:
            kyc_complete = None

        # Relationship manager assigned for HNI/senior
        rm_assigned = segment in ["hni", "salaried_senior"]

        # Credit score (CIBIL range: 300-900)
        if segment == "hni":
            cibil = random.randint(750, 900)
        elif segment in ["salaried_mid", "salaried_senior"]:
            cibil = random.randint(650, 850)
        elif segment == "student":
            cibil = random.randint(0, 600)
        else:
            cibil = random.randint(550, 800)
        if random.random() < 0.08:
            cibil = None  # no credit history

        # Digital adoption score (0-10)
        digital_score = round(
            random.uniform(3, 10) if segment != "retired"
            else random.uniform(1, 6), 1
        )

        # NRI flag
        is_nri = random.random() < 0.03

        # Gender
        gender = random.choices(
            ["Male", "Female", "Other"],
            weights=[0.55, 0.44, 0.01]
        )[0]
        if random.random() < 0.02:
            gender = None

        records.append({
            "customer_id"       : customer_id,
            "age"               : age,
            "gender"            : gender,
            "city"              : city,
            "city_tier"         : tier,
            "segment"           : segment,
            "monthly_income"    : monthly_income,
            "account_open_date" : account_open.isoformat(),
            "account_age_days"  : account_age_days,
            "kyc_complete"      : kyc_complete,
            "pan_linked"        : pan_linked,
            "aadhaar_linked"    : aadhaar_linked,
            "cibil_score"       : cibil,
            "digital_score"     : digital_score,
            "rm_assigned"       : rm_assigned,
            "is_nri"            : is_nri,
            "occupation"        : segment.replace("_", " ").title(),
            "phone"             : fake.phone_number()
                                  if random.random() > 0.02 else None,
            "email"             : fake.email()
                                  if random.random() > 0.05 else None,
        })

    df = pd.DataFrame(records)
    # Inject duplicates
    dupes = df.sample(frac=0.005, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)

    print(f"  → {len(df):,} rows | "
          f"{df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  TABLE 2 — PRODUCTS
# ================================================================

def generate_products(customers_df):
    print("Generating products table...")
    records = []

    # Product eligibility by segment
    SEGMENT_PRODUCTS = {
        "student"        : ["savings_account", "recurring_deposit"],
        "salaried_entry" : ["savings_account", "recurring_deposit",
                            "mutual_fund", "credit_card",
                            "personal_loan"],
        "salaried_mid"   : ["savings_account", "fixed_deposit",
                            "mutual_fund", "credit_card",
                            "home_loan", "personal_loan",
                            "insurance"],
        "salaried_senior": ["savings_account", "fixed_deposit",
                            "mutual_fund", "credit_card",
                            "home_loan", "insurance"],
        "self_employed"  : ["savings_account", "fixed_deposit",
                            "mutual_fund", "credit_card",
                            "personal_loan", "insurance"],
        "hni"            : ["savings_account", "fixed_deposit",
                            "mutual_fund", "credit_card",
                            "home_loan", "personal_loan",
                            "insurance"],
        "retired"        : ["savings_account", "fixed_deposit",
                            "recurring_deposit", "insurance"],
    }

    customers_clean = customers_df.drop_duplicates(
        subset=["customer_id"]
    )

    for _, cust in customers_clean.iterrows():
        cust_id = cust["customer_id"]
        segment = cust["segment"]
        income  = cust["monthly_income"] or 30000

        eligible = SEGMENT_PRODUCTS.get(segment, ["savings_account"])

        # Every customer has a savings account
        records.append({
            "product_id"   : f"PROD_{cust_id}_SA",
            "customer_id"  : cust_id,
            "product_type" : "savings_account",
            "open_date"    : cust["account_open_date"],
            "current_value": round(
                random.uniform(1000, income * 3), 2
            ),
            "is_active"    : True,
            "interest_rate": 3.5,
        })

        # Additional products based on segment and probability
        product_probs = {
            "fixed_deposit"    : 0.40,
            "recurring_deposit": 0.25,
            "mutual_fund"      : 0.30,
            "credit_card"      : 0.45,
            "home_loan"        : 0.20,
            "personal_loan"    : 0.25,
            "insurance"        : 0.35,
        }

        for ptype in eligible[1:]:  # skip savings (already added)
            prob = product_probs.get(ptype, 0.20)
            if segment == "hni":
                prob = min(prob * 1.5, 0.85)
            elif segment == "student":
                prob = min(prob * 0.5, 0.30)

            if random.random() < prob:
                # Compute realistic product value
                if ptype == "home_loan":
                    value = round(
                        income * random.uniform(40, 80), 2
                    )
                elif ptype == "personal_loan":
                    value = round(
                        income * random.uniform(5, 20), 2
                    )
                elif ptype == "credit_card":
                    value = round(
                        income * random.uniform(2, 5), 2
                    )  # credit limit
                elif ptype == "mutual_fund":
                    value = round(
                        income * random.uniform(3, 30), 2
                    )
                elif ptype == "fixed_deposit":
                    value = round(
                        income * random.uniform(6, 36), 2
                    )
                elif ptype == "insurance":
                    value = round(
                        income * random.uniform(0.5, 2), 2
                    )  # annual premium
                else:
                    value = round(
                        income * random.uniform(1, 6), 2
                    )

                open_dt = random_date(
                    date(2019, 1, 1), date(2023, 6, 30)
                )
                is_active = random.random() > 0.10

                records.append({
                    "product_id"   : f"PROD_{cust_id}_{ptype[:3].upper()}",
                    "customer_id"  : cust_id,
                    "product_type" : ptype,
                    "open_date"    : open_dt.isoformat(),
                    "current_value": value,
                    "is_active"    : is_active,
                    "interest_rate": PRODUCT_TYPES[ptype].get(
                        "interest_rate", 0
                    ),
                })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | "
          f"{df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  TABLE 3 — TRANSACTIONS
# ================================================================

def generate_transactions(customers_df, start_date, end_date,
                           table_suffix="history", conn=None,
                           table_name=None):
    """
    Generate transaction records for all customers
    between start_date and end_date.

    Transactions are streamed directly to SQLite so the
    process avoids allocating a giant in-memory pandas frame.
    """
    print(f"Generating transactions_{table_suffix} table...")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Estimated rows: ~{N_CUSTOMERS * 5 * 12:,}")

    if conn is not None:
        table_name = table_name or f"transactions_{table_suffix}"
        create_transactions_table(conn, table_name)

    records = []
    txn_counter = 0
    total_rows = 0

    customers_clean = customers_df.drop_duplicates(
        subset=["customer_id"]
    ).reset_index(drop=True)

    current_month = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)

    months = []
    m = current_month
    while m <= end_month:
        months.append(m)
        if m.month == 12:
            m = date(m.year + 1, 1, 1)
        else:
            m = date(m.year, m.month + 1, 1)

    for customer_idx, cust in enumerate(customers_clean.itertuples(), start=1):
        cust_id = cust.customer_id
        segment = cust.segment
        income = cust.monthly_income or 30000
        tier = cust.city_tier
        digital = cust.digital_score or 5

        base_txn_count = monthly_txn_count(segment, tier)
        balance = round(income * random.uniform(0.5, 3.0), 2)

        for month_start in months:
            if random.random() < 0.03:
                continue

            n_txns = max(1, int(base_txn_count * random.uniform(0.7, 1.3)))

            if segment not in ["student", "retired"]:
                salary_date = date(
                    month_start.year, month_start.month,
                    random.randint(1, 7)
                )
                balance += income
                txn_counter += 1
                records.append({
                    "txn_id": f"TXN{txn_counter:010d}",
                    "customer_id": cust_id,
                    "txn_date": salary_date.isoformat(),
                    "txn_type": "NetBanking",
                    "amount": income,
                    "credit_debit": "CR",
                    "merchant_category": "salary_credit",
                    "channel": "netbanking",
                    "balance_after": round(balance, 2),
                    "city": cust.city,
                    "is_international": False,
                })

            for _ in range(n_txns):
                txn_weights = []
                for ttype, tinfo in TXN_TYPES.items():
                    w = tinfo["weight"]
                    if ttype == "UPI" and digital > 7:
                        w *= 1.5
                    elif ttype == "Branch" and digital < 4:
                        w *= 2.0
                    txn_weights.append(w)

                txn_type = random.choices(
                    list(TXN_TYPES.keys()),
                    weights=txn_weights
                )[0]
                txn_info = TXN_TYPES[txn_type]

                base_amt = txn_info["avg_amount"]
                amount = round(
                    abs(np.random.lognormal(np.log(base_amt), 0.8)), 2
                )
                amount = max(1.0, min(amount, income * 10))

                if month_start.month == 12:
                    next_month = date(month_start.year + 1, 1, 1)
                else:
                    next_month = date(month_start.year, month_start.month + 1, 1)
                txn_date = random_date(
                    month_start,
                    min(next_month - timedelta(days=1), end_date)
                )

                credit_debit = "CR" if txn_type in ["NetBanking"] and random.random() < 0.2 else "DR"

                if credit_debit == "DR":
                    balance = max(0, balance - amount)
                else:
                    balance += amount

                merchant_cat = random.choices(
                    MERCHANT_CATEGORIES,
                    weights=[0.15, 0.12, 0.08, 0.07,
                             0.06, 0.06, 0.08, 0.07,
                             0.10, 0.05, 0.04, 0.05,
                             0.03, 0.02, 0.01, 0.01]
                )[0]

                channel = random.choices(
                    CHANNELS,
                    weights=[0.40, 0.20, 0.15,
                             0.10, 0.05, 0.10]
                )[0]

                if random.random() < 0.02:
                    merchant_cat = None
                if random.random() < 0.015:
                    channel = None

                txn_counter += 1
                records.append({
                    "txn_id": f"TXN{txn_counter:010d}",
                    "customer_id": cust_id,
                    "txn_date": txn_date.isoformat(),
                    "txn_type": txn_type,
                    "amount": amount,
                    "credit_debit": credit_debit,
                    "merchant_category": merchant_cat,
                    "channel": channel,
                    "balance_after": round(balance, 2),
                    "city": cust.city,
                    "is_international": random.random() < 0.02,
                })

        if customer_idx % 10_000 == 0:
            print(f"    Processed {customer_idx:,} customers, "
                  f"{len(records):,} transactions buffered...")

        if len(records) >= 10_000:
            if conn is not None:
                insert_records_to_sqlite(conn, table_name, records)
                total_rows += len(records)
                records.clear()

    if records:
        if conn is not None:
            insert_records_to_sqlite(conn, table_name, records)
            total_rows += len(records)
            records.clear()

    if conn is not None:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  → {row_count:,} rows")
        return None

    return None


# ================================================================
#  TABLE 4 — CLV LABELS
# ================================================================

def generate_clv_labels(customers_df, products_df,
                         label_transactions_df=None, conn=None,
                         table_name="transactions_label"):
    """
    Compute actual CLV for each customer over the label period
    (Jan 2024 - Dec 2024).

    This is the TARGET VARIABLE — what we want to predict
    using only information available at the end of 2023.
    """
    print("Computing CLV labels...")

    customers_clean = customers_df.drop_duplicates(
        subset=["customer_id"]
    )

    # Index products for fast lookup
    products_by_cust = products_df.groupby("customer_id")

    records = []
    for _, cust in customers_clean.iterrows():
        cust_id = cust["customer_id"]

        # Get this customer's products
        prods = products_by_cust.get_group(cust_id) \
            if cust_id in products_by_cust.groups else \
            pd.DataFrame()

        if label_transactions_df is not None:
            label_txns = label_transactions_df.loc[
                label_transactions_df["customer_id"] == cust_id
            ].copy()
        elif conn is not None:
            label_txns = pd.read_sql_query(
                f"SELECT * FROM {table_name} WHERE customer_id = ?",
                conn,
                params=(cust_id,),
            )
        else:
            label_txns = pd.DataFrame()

        clv = compute_clv(cust, prods, label_txns)

        # Segment-based adjustment to ensure realistic distribution
        segment = cust["segment"]
        if segment == "hni":
            clv = max(clv, random.uniform(50000, 500000))
        elif segment == "salaried_senior":
            clv = max(clv, random.uniform(10000, 80000))
        elif segment == "student":
            clv = min(clv, random.uniform(500, 3000))

        # Add more zeros/near-zeros for realism
        if random.random() < 0.05:
            clv = round(random.uniform(0, 200), 2)

        records.append({
            "customer_id"         : cust_id,
            "clv_next_12months"   : round(clv, 2),
            "clv_bucket"          : (
                "very_low"   if clv < 1000 else
                "low"        if clv < 5000 else
                "medium"     if clv < 20000 else
                "high"       if clv < 100000 else
                "very_high"
            ),
            "label_period_start"  : LABEL_START.isoformat(),
            "label_period_end"    : LABEL_END.isoformat(),
        })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows")
    print(f"\n  CLV Distribution:")
    print(f"    Mean   : ₹{df['clv_next_12months'].mean():>10,.2f}")
    print(f"    Median : ₹{df['clv_next_12months'].median():>10,.2f}")
    print(f"    Std    : ₹{df['clv_next_12months'].std():>10,.2f}")
    print(f"    Min    : ₹{df['clv_next_12months'].min():>10,.2f}")
    print(f"    Max    : ₹{df['clv_next_12months'].max():>10,.2f}")

    print(f"\n  CLV Bucket Distribution:")
    bucket_counts = df["clv_bucket"].value_counts()
    for bucket, count in bucket_counts.items():
        pct = count / len(df) * 100
        print(f"    {bucket:<12}: {count:>7,}  ({pct:.1f}%)")

    return df


# ================================================================
#  WRITE TO SQLITE
# ================================================================

def write_to_sqlite(tables: dict, db_path: str, conn=None):
    """
    Write all DataFrames to SQLite tables.
    Creates indexes on primary and foreign keys for
    fast PySpark JDBC reads.
    """
    print(f"\nWriting to SQLite database: {db_path}")

    if conn is None:
        conn = sqlite3.connect(db_path)

    for table_name, df in tables.items():
        print(f"  Writing {table_name}... ", end="")
        df.to_sql(
            table_name, conn,
            if_exists="replace",
            index=False,
            chunksize=10_000
        )
        print(f"{len(df):,} rows ✅")

    # Create indexes for fast joins
    print("\n  Creating indexes...")
    indexes = [
        "CREATE INDEX idx_cust_id ON customers(customer_id)",
        "CREATE INDEX idx_prod_cust ON products(customer_id)",
        "CREATE INDEX idx_txn_hist_cust ON transactions_history(customer_id)",
        "CREATE INDEX idx_txn_hist_date ON transactions_history(txn_date)",
        "CREATE INDEX idx_txn_label_cust ON transactions_label(customer_id)",
        "CREATE INDEX idx_clv_cust ON clv_labels(customer_id)",
    ]

    for idx_sql in indexes:
        try:
            conn.execute(idx_sql)
            print(f"    ✅ {idx_sql.split('ON')[1].strip()}")
        except Exception as e:
            print(f"    ⚠️ {e}")

    conn.commit()

    if conn is not None and os.path.exists(db_path):
        conn.close()

    # File size
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\n  Database file size: {size_mb:.1f} MB")


# ================================================================
#  MAIN
# ================================================================

def main():
    print("=" * 65)
    print(" RealWorld-DS-ML | Finance | CLV Prediction")
    print(" Country        : India")
    print(" Context        : Indian retail banking")
    print(" History period : Jan 2023 - Dec 2023")
    print(" Label period   : Jan 2024 - Dec 2024")
    print(" Problem type   : Regression")
    print(" Target         : clv_next_12months (₹)")
    print(" Processing     : PySpark + SQLite")
    print("=" * 65)
    print(f" Customers      : {N_CUSTOMERS:,}")
    print(f" Expected txns  : ~{N_CUSTOMERS * 5 * 12:,}")
    print(f" Database       : {os.path.abspath(DB_PATH)}")
    print("=" * 65)

    # ── Generate all tables ───────────────────────────────────
    target_db_path = DB_PATH
    if os.path.exists(DB_PATH):
        try:
            probe_conn = sqlite3.connect(DB_PATH, timeout=2.0)
            probe_conn.execute("SELECT 1")
            probe_conn.close()
        except sqlite3.OperationalError:
            target_db_path = DB_FALLBACK_PATH
            print(f"  Using alternate database path: {target_db_path}")

    conn = sqlite3.connect(target_db_path, timeout=60.0)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = DELETE")

    try:
        customers_df = generate_customers()
        products_df  = generate_products(customers_df)

        print("\nGenerating history transactions (2023)...")
        generate_transactions(
            customers_df,
            HISTORY_START, HISTORY_END,
            table_suffix="history",
            conn=conn,
            table_name="transactions_history",
        )

        print("\nGenerating label transactions (2024)...")
        generate_transactions(
            customers_df,
            LABEL_START, LABEL_END,
            table_suffix="label",
            conn=conn,
            table_name="transactions_label",
        )

        print("\nComputing CLV labels...")
        clv_labels_df = generate_clv_labels(
            customers_df, products_df, conn=conn,
            table_name="transactions_label"
        )

        # ── Write to SQLite ─────────────────────────────────────
        tables = {
            "customers": customers_df,
            "products": products_df,
            "clv_labels": clv_labels_df,
        }

        write_to_sqlite(tables, target_db_path, conn=conn)
    finally:
        conn.close()

    # ── Final summary ─────────────────────────────────────────
    print()
    print("=" * 65)
    print(" ✅ Database ready at data/finance_clv.db")
    print("=" * 65)
    print(f"  customers            : {len(customers_df):>8,} rows")
    print(f"  products             : {len(products_df):>8,} rows")
    print(f"  clv_labels           : {len(clv_labels_df):>8,} rows")
    print("=" * 65)
    print()
    print(" How to read in PySpark (Notebook 01):")
    print("""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \\
        .appName("CLV") \\
        .config("spark.jars", "sqlite-jdbc.jar") \\
        .getOrCreate()

    df = spark.read.format("jdbc") \\
        .option("url", "jdbc:sqlite:../data/finance_clv.db") \\
        .option("dbtable", "customers") \\
        .load()
    """)
    print("=" * 65)
    print()
    print(" Next step → Open notebooks/01_database_setup.ipynb")
    print("=" * 65)


if __name__ == "__main__":
    main()