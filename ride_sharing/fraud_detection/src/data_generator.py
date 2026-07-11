"""
===============================================================
 RealWorld-DS-ML | Ride Sharing | Fraud Detection
 File    : data_generator.py
 Purpose : Generate realistic, production-style data simulating
           ride transaction fraud on a Bangalore ride sharing
           platform (Ola/Uber/Rapido style).

 Tables Generated:
   1. transactions.csv — main fact table, one row per ride
                         transaction, target variable lives here
   2. drivers.csv      — driver profiles with behavioral history
   3. users.csv        — user profiles with behavioral history
   4. devices.csv      — device fingerprint signals

 Target Variable (Stage 1 — Binary):
   is_fraud: 0 = legitimate, 1 = fraud

 Target Variable (Stage 2 — Multi-class, fraud rows only):
   fraud_label:
     0 = legitimate
     1 = driver_fraud    (GPS spoofing, fake completions)
     2 = user_fraud      (payment fraud, promo abuse)
     3 = collusion_fraud (driver + user working together)

 Class Distribution (realistic extreme imbalance):
   Legitimate      : ~97.0%
   Driver fraud    :  ~1.2%
   User fraud      :  ~1.0%
   Collusion fraud :  ~0.8%
   Total fraud     :  ~3.0%

 Key Design Decisions:
   - Fraud is deliberately designed to look almost like
     legitimate transactions — fraudsters try to avoid detection
   - Velocity features: rapid repeated transactions = suspicious
   - GPS integrity: spoofed GPS leaves subtle traces
   - Network features: collusion shows unusual pair frequency
   - Device signals: fraud accounts share devices or use VPNs
   - Intentional messiness: nulls, outliers, duplicates
===============================================================
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import math
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

# ── Config ───────────────────────────────────────────────────
N_TRANSACTIONS = 60_000
N_DRIVERS      = 3_000
N_USERS        = 20_000
N_DEVICES      = 22_000   # more devices than users — shared devices

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Bangalore Zones ──────────────────────────────────────────
BANGALORE_ZONES = {
    "Koramangala"       : {"lat": (12.926, 12.940), "lon": (77.614, 77.632)},
    "Whitefield"        : {"lat": (12.960, 12.990), "lon": (77.730, 77.760)},
    "Indiranagar"       : {"lat": (12.970, 12.985), "lon": (77.635, 77.650)},
    "HSR Layout"        : {"lat": (12.905, 12.925), "lon": (77.630, 77.650)},
    "Electronic City"   : {"lat": (12.830, 12.865), "lon": (77.660, 77.690)},
    "Marathahalli"      : {"lat": (12.945, 12.965), "lon": (77.695, 77.715)},
    "Bellandur"         : {"lat": (12.915, 12.935), "lon": (77.665, 77.685)},
    "Sarjapur Road"     : {"lat": (12.880, 12.915), "lon": (77.670, 77.700)},
    "BTM Layout"        : {"lat": (12.910, 12.928), "lon": (77.608, 77.625)},
    "Jayanagar"         : {"lat": (12.920, 12.940), "lon": (77.575, 77.595)},
    "JP Nagar"          : {"lat": (12.890, 12.915), "lon": (77.570, 77.595)},
    "Banashankari"      : {"lat": (12.895, 12.915), "lon": (77.545, 77.570)},
    "Rajajinagar"       : {"lat": (12.985, 13.005), "lon": (77.545, 77.565)},
    "Malleswaram"       : {"lat": (13.000, 13.020), "lon": (77.560, 77.580)},
    "Hebbal"            : {"lat": (13.030, 13.055), "lon": (77.585, 77.610)},
    "Yelahanka"         : {"lat": (13.090, 13.120), "lon": (77.590, 77.620)},
    "Yeshwanthpur"      : {"lat": (13.015, 13.035), "lon": (77.535, 77.560)},
    "Cunningham Road"   : {"lat": (12.990, 13.005), "lon": (77.590, 77.610)},
    "MG Road"           : {"lat": (12.970, 12.985), "lon": (77.600, 77.620)},
    "Shivajinagar"      : {"lat": (12.982, 12.998), "lon": (77.595, 77.615)},
    "KR Puram"          : {"lat": (13.000, 13.025), "lon": (77.685, 77.710)},
    "Bannerghatta Road" : {"lat": (12.855, 12.890), "lon": (77.580, 77.610)},
    "Kadugodi"          : {"lat": (12.975, 12.998), "lon": (77.755, 77.780)},
    "Nagawara"          : {"lat": (13.040, 13.060), "lon": (77.615, 77.635)},
    "Vijayanagar"       : {"lat": (12.965, 12.985), "lon": (77.520, 77.545)},
}

ZONE_NAMES = list(BANGALORE_ZONES.keys())

# ── India Public Holidays 2025 ────────────────────────────────
INDIA_HOLIDAYS_2025 = {
    "2025-01-01", "2025-01-14", "2025-01-26",
    "2025-02-26", "2025-03-14", "2025-03-31",
    "2025-04-06", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-12", "2025-06-07",
    "2025-07-06", "2025-08-15", "2025-08-16",
    "2025-09-05", "2025-10-02", "2025-10-20",
    "2025-10-21", "2025-11-01", "2025-11-05",
    "2025-12-25",
}

MONSOON_MONTHS  = {6, 7, 8, 9}
VEHICLE_TYPES   = ["Auto", "Mini", "Sedan", "SUV", "Bike"]
VEHICLE_WEIGHTS = [0.20,   0.30,   0.25,   0.15,  0.10]

PAYMENT_METHODS    = ["UPI", "Cash", "Card", "Wallet"]
PAYMENT_WEIGHTS    = [0.50,  0.25,   0.15,   0.10]

# ── Fraud type labels ─────────────────────────────────────────
LEGIT      = 0
DRIVER_FRAUD    = 1
USER_FRAUD      = 2
COLLUSION_FRAUD = 3

# ── Fraud rates ───────────────────────────────────────────────
FRAUD_RATE_DRIVER    = 0.012   # 1.2%
FRAUD_RATE_USER      = 0.010   # 1.0%
FRAUD_RATE_COLLUSION = 0.008   # 0.8%
# Total fraud: ~3.0%


# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def random_timestamp_2025():
    """Random datetime within full year 2025."""
    start = datetime(2025, 1, 1, 0, 0, 0)
    end   = datetime(2025, 12, 31, 23, 59, 59)
    delta = end - start
    return start + timedelta(
        seconds=random.randint(0, int(delta.total_seconds()))
    )


def haversine_distance(lat1, lon1, lat2, lon2):
    """Straight-line distance between two GPS points in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def compute_expected_fare(vehicle_type, distance_km, surge):
    """Calculate expected fare — used to detect fare anomalies."""
    base_fare   = {"Auto": 30, "Mini": 50, "Sedan": 70, "SUV": 100, "Bike": 20}
    per_km_rate = {"Auto": 12, "Mini": 14, "Sedan": 17, "SUV":  22, "Bike": 8}
    return round(
        (base_fare[vehicle_type] + per_km_rate[vehicle_type] * distance_km)
        * surge, 2
    )


# ================================================================
#  TABLE 1 — DEVICES
# ================================================================

def generate_devices():
    """
    Device fingerprint table. Each device has signals that help
    identify suspicious behavior. Some devices are shared across
    multiple accounts — a strong fraud signal.
    """
    print("Generating devices table...")
    records = []

    os_types      = ["Android", "iOS", "Android", "Android", "iOS"]
    device_models = [
        "Samsung Galaxy A53", "iPhone 14", "Redmi Note 11",
        "OnePlus Nord CE3", "Vivo Y73", "Realme 9 Pro",
        "Samsung Galaxy M34", "iPhone 12", "POCO X5 Pro", None
    ]

    for i in range(N_DEVICES):
        device_id = f"DEV{str(i+1).zfill(6)}"

        # Most devices are clean — small % are suspicious
        device_risk = random.choices(
            ["clean", "moderate", "high"],
            weights=[0.88, 0.08, 0.04]
        )[0]

        # VPN usage — fraudsters often use VPNs to hide location
        uses_vpn = False
        if device_risk == "high":
            uses_vpn = random.random() < 0.65
        elif device_risk == "moderate":
            uses_vpn = random.random() < 0.20

        # Number of accounts linked to this device
        # Clean devices have 1 account, suspicious have multiple
        if device_risk == "clean":
            accounts_linked = 1
        elif device_risk == "moderate":
            accounts_linked = random.randint(1, 3)
        else:
            accounts_linked = random.randint(3, 8)

        # GPS spoofing capability — high risk devices may use mock GPS
        gps_mock_detected = False
        if device_risk == "high":
            gps_mock_detected = random.random() < 0.45

        # Device age in days
        device_age_days = random.randint(1, 1500)
        if device_risk == "high":
            device_age_days = random.randint(1, 60)  # new devices = riskier

        records.append({
            "device_id"          : device_id,
            "os_type"            : random.choice(os_types),
            "device_model"       : random.choice(device_models),
            "device_age_days"    : device_age_days,
            "accounts_linked"    : accounts_linked,
            "uses_vpn"           : uses_vpn,
            "gps_mock_detected"  : gps_mock_detected,
            "device_risk_level"  : device_risk,
            "is_rooted_jailbreak": (device_risk == "high" and
                                    random.random() < 0.30),
            "app_version"        : random.choice([
                "3.2.1", "3.3.0", "3.3.1", "3.4.0", "2.9.8", None
            ]),
        })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  TABLE 2 — DRIVERS
# ================================================================

def generate_drivers():
    print("Generating drivers table...")
    records = []

    for i in range(N_DRIVERS):
        driver_id   = f"DRV{str(i+1).zfill(5)}"
        joined_date = datetime(2025, 1, 1) - timedelta(
            days=random.randint(30, 2500)
        )

        # Driver fraud propensity — most are honest
        fraud_propensity = random.choices(
            ["honest", "opportunistic", "habitual_fraud"],
            weights=[0.88, 0.08, 0.04]
        )[0]

        rating = round(np.random.normal(4.1, 0.4), 2)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.03:
            rating = None

        vehicle_type = random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0]

        # Historical fraud indicators
        if fraud_propensity == "honest":
            past_fraud_flags        = 0
            gps_anomaly_count       = random.randint(0, 2)
            avg_daily_rides         = round(random.uniform(6, 14), 1)
            completion_rate         = round(random.uniform(0.88, 1.00), 2)
            unusual_route_pct       = round(random.uniform(0.0, 0.05), 3)
        elif fraud_propensity == "opportunistic":
            past_fraud_flags        = random.randint(1, 3)
            gps_anomaly_count       = random.randint(3, 10)
            avg_daily_rides         = round(random.uniform(12, 22), 1)
            completion_rate         = round(random.uniform(0.80, 0.92), 2)
            unusual_route_pct       = round(random.uniform(0.05, 0.15), 3)
        else:
            past_fraud_flags        = random.randint(4, 15)
            gps_anomaly_count       = random.randint(10, 40)
            avg_daily_rides         = round(random.uniform(18, 35), 1)
            completion_rate         = round(random.uniform(0.70, 0.85), 2)
            unusual_route_pct       = round(random.uniform(0.15, 0.40), 3)

        exp_years = round(
            (datetime(2025, 12, 31) - joined_date).days / 365, 1
        )

        records.append({
            "driver_id"              : driver_id,
            "name"                   : fake.name(),
            "phone"                  : fake.phone_number() if random.random() > 0.02 else None,
            "vehicle_type"           : vehicle_type,
            "vehicle_model"          : random.choice([
                "Maruti Swift", "Hyundai i20", "Honda City",
                "Bajaj RE Auto", "TVS Jupiter", None
            ]),
            "rating"                 : rating,
            "fraud_propensity"       : fraud_propensity,
            "past_fraud_flags"       : past_fraud_flags,
            "gps_anomaly_count"      : gps_anomaly_count,
            "avg_daily_rides"        : avg_daily_rides,
            "completion_rate"        : completion_rate,
            "unusual_route_pct"      : unusual_route_pct,
            "joined_date"            : joined_date.strftime("%Y-%m-%d"),
            "experience_years"       : exp_years,
            "total_rides"            : random.randint(50, 12000),
            "is_active"              : random.choice([True, False]),
            "home_zone"              : random.choice(ZONE_NAMES),
            "device_id"              : f"DEV{str(random.randint(1, N_DEVICES)).zfill(6)}",
        })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  TABLE 3 — USERS
# ================================================================

def generate_users():
    print("Generating users table...")
    records = []

    for i in range(N_USERS):
        user_id = f"USR{str(i+1).zfill(6)}"
        signup  = datetime(2025, 1, 1) - timedelta(
            days=random.randint(0, 1460)
        )

        # User fraud propensity
        fraud_propensity = random.choices(
            ["honest", "opportunistic", "habitual_fraud"],
            weights=[0.90, 0.07, 0.03]
        )[0]

        age = random.randint(18, 60)
        if random.random() < 0.02:
            age = random.choice([-1, 0, 150, 999])

        rating = round(np.random.normal(4.2, 0.5), 1)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.04:
            rating = None

        account_age_days = (datetime(2025, 1, 1) - signup).days

        if fraud_propensity == "honest":
            past_chargebacks      = 0
            promo_abuse_count     = random.randint(0, 1)
            multiple_accounts     = False
            avg_weekly_rides      = round(random.uniform(1, 8), 1)
            payment_failure_rate  = round(random.uniform(0.0, 0.05), 3)
        elif fraud_propensity == "opportunistic":
            past_chargebacks      = random.randint(1, 3)
            promo_abuse_count     = random.randint(2, 6)
            multiple_accounts     = random.random() < 0.30
            avg_weekly_rides      = round(random.uniform(5, 15), 1)
            payment_failure_rate  = round(random.uniform(0.05, 0.15), 3)
        else:
            past_chargebacks      = random.randint(3, 12)
            promo_abuse_count     = random.randint(6, 20)
            multiple_accounts     = random.random() < 0.75
            avg_weekly_rides      = round(random.uniform(12, 30), 1)
            payment_failure_rate  = round(random.uniform(0.15, 0.40), 3)

        records.append({
            "user_id"              : user_id,
            "name"                 : fake.name(),
            "phone"                : fake.phone_number() if random.random() > 0.03 else None,
            "email"                : fake.email() if random.random() > 0.05 else None,
            "age"                  : age,
            "gender"               : random.choice(["Male","Female","Other",None]),
            "signup_date"          : signup.strftime("%Y-%m-%d"),
            "account_age_days"     : account_age_days,
            "rating"               : rating,
            "fraud_propensity"     : fraud_propensity,
            "past_chargebacks"     : past_chargebacks,
            "promo_abuse_count"    : promo_abuse_count,
            "multiple_accounts"    : multiple_accounts,
            "avg_weekly_rides"     : avg_weekly_rides,
            "payment_failure_rate" : payment_failure_rate,
            "total_rides"          : random.randint(0, 400),
            "is_prime_member"      : random.choice([True, False, None]),
            "home_zone"            : random.choice(ZONE_NAMES),
            "device_id"            : f"DEV{str(random.randint(1, N_DEVICES)).zfill(6)}",
        })

    df = pd.DataFrame(records)
    # ~1% duplicate users
    dupes = df.sample(frac=0.01, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)

    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  FRAUD GENERATION FUNCTIONS
# ================================================================

def generate_legitimate_signals(
    distance_km, vehicle_type, surge, ts, hour
):
    """
    Generate GPS and timing signals for a LEGITIMATE transaction.
    These look clean and consistent.
    """
    # Legitimate completion time is physically plausible
    # distance / realistic_speed_kmph × 60 minutes
    base_speed   = {"Auto":18,"Mini":22,"Sedan":25,"SUV":28,"Bike":24}[vehicle_type]
    traffic_adj  = random.uniform(0.6, 1.0) if hour in range(8,21) else 1.0
    completion_min = round(
        (distance_km / (base_speed * traffic_adj)) * 60
        + random.uniform(-2, 3), 1
    )
    completion_min = max(1.0, completion_min)

    # Distance moved by driver is close to expected
    distance_moved_km = round(
        distance_km * random.uniform(0.85, 1.15), 3
    )

    # GPS signal is realistic — not perfectly clean
    gps_signal_strength = round(random.uniform(0.70, 0.95), 2)
    gps_points_recorded = random.randint(
        int(completion_min * 4), int(completion_min * 6)
    )

    # Route deviation is low
    route_deviation_score = round(random.uniform(0.0, 0.15), 3)

    # Speed is plausible
    if completion_min > 0:
        claimed_speed_kmph = round(
            (distance_km / (completion_min / 60)), 1
        )
    else:
        claimed_speed_kmph = 20.0

    return {
        "completion_time_min"    : completion_min,
        "distance_moved_km"      : distance_moved_km,
        "gps_signal_strength"    : gps_signal_strength,
        "gps_points_recorded"    : gps_points_recorded,
        "route_deviation_score"  : route_deviation_score,
        "claimed_speed_kmph"     : claimed_speed_kmph,
    }


def generate_driver_fraud_signals(
    distance_km, vehicle_type, surge, ts, hour
):
    """
    Generate signals for DRIVER FRAUD.
    Driver marks trip as complete without actually driving.
    GPS is spoofed — driver stays near pickup or moves minimally.

    Key fraud signatures:
    - distance_moved << claimed distance
    - GPS signal too perfect (spoofing apps are too clean)
    - Completion time too fast for distance
    - GPS points recorded suspiciously round number
    """
    # Claimed distance is normal — driver fakes the trip
    # But actual movement is very small
    distance_moved_km = round(
        distance_km * random.uniform(0.02, 0.20), 3
    )  # only moved 2-20% of claimed distance

    # Completion time is suspiciously fast
    # A 10km trip "completed" in 4 minutes = impossible
    completion_min = round(
        (distance_km / 40) * 60 * random.uniform(0.15, 0.40), 1
    )
    completion_min = max(0.5, completion_min)

    # GPS spoofing apps produce unnaturally clean signal
    gps_signal_strength = round(random.uniform(0.96, 1.00), 2)

    # Spoofed GPS often records exact round intervals
    gps_points_recorded = random.choice([
        10, 12, 15, 20, 25, 30
    ])

    # Route deviation is high — spoofed path doesn't follow real roads
    route_deviation_score = round(random.uniform(0.40, 0.90), 3)

    # Claimed speed is impossibly high
    if completion_min > 0:
        claimed_speed_kmph = round(
            (distance_km / (completion_min / 60)), 1
        )
    else:
        claimed_speed_kmph = 999.0

    return {
        "completion_time_min"    : completion_min,
        "distance_moved_km"      : distance_moved_km,
        "gps_signal_strength"    : gps_signal_strength,
        "gps_points_recorded"    : gps_points_recorded,
        "route_deviation_score"  : route_deviation_score,
        "claimed_speed_kmph"     : claimed_speed_kmph,
    }


def generate_user_fraud_signals(
    distance_km, vehicle_type, surge, ts, hour
):
    """
    Generate signals for USER FRAUD.
    Stolen payment methods, promo abuse, chargeback fraud.

    These trips often COMPLETE legitimately in terms of GPS —
    the fraud is in the payment, not the ride itself.
    Key signatures:
    - New account (account_age_days very low)
    - Payment method recently added
    - Unusual payment method for this user profile
    - Transaction at unusual hour
    - Multiple promo codes attempted
    """
    # GPS is mostly clean — the ride actually happened
    # but payment is fraudulent
    base_signals = generate_legitimate_signals(
        distance_km, vehicle_type, surge, ts, hour
    )

    # Slight modifications — user fraud rides tend to be longer
    # (fraudsters maximize value of stolen cards)
    base_signals["completion_time_min"] = round(
        base_signals["completion_time_min"] * random.uniform(1.0, 1.3), 1
    )

    return base_signals


def generate_collusion_fraud_signals(
    distance_km, vehicle_type, surge, ts, hour
):
    """
    Generate signals for COLLUSION FRAUD.
    Driver and user are working together. Both accounts are
    controlled by the same fraudster or a network.

    They run fake rides to:
    - Generate referral bonuses
    - Manipulate driver payout systems
    - Launder money through the platform

    Key signatures:
    - Driver and user have ridden together many times before
    - Both accounts created around the same time
    - Rides always happen at the same times of day
    - Distance is always suspiciously similar ride to ride
    - GPS moves but the route is always identical
    """
    # GPS moves — unlike driver fraud, collusion trips often
    # actually drive somewhere (to appear legitimate)
    # but distance is oddly consistent
    distance_moved_km = round(
        distance_km * random.uniform(0.75, 0.95), 3
    )

    # Completion time is slightly fast but not impossible
    completion_min = round(
        (distance_km / 25) * 60 * random.uniform(0.60, 0.85), 1
    )
    completion_min = max(1.0, completion_min)

    # GPS is clean — they actually drive
    gps_signal_strength = round(random.uniform(0.80, 0.95), 2)
    gps_points_recorded = random.randint(
        int(completion_min * 3), int(completion_min * 5)
    )

    # Route deviation is moderate — they drive but on identical routes
    route_deviation_score = round(random.uniform(0.10, 0.30), 3)

    if completion_min > 0:
        claimed_speed_kmph = round(
            (distance_km / (completion_min / 60)), 1
        )
    else:
        claimed_speed_kmph = 30.0

    return {
        "completion_time_min"    : completion_min,
        "distance_moved_km"      : distance_moved_km,
        "gps_signal_strength"    : gps_signal_strength,
        "gps_points_recorded"    : gps_points_recorded,
        "route_deviation_score"  : route_deviation_score,
        "claimed_speed_kmph"     : claimed_speed_kmph,
    }


# ================================================================
#  TABLE 4 — TRANSACTIONS  (main fact table)
# ================================================================

def generate_transactions(drivers_df, users_df, devices_df):
    print("Generating transactions table... this may take a moment")
    records = []

    # Build fast lookup dicts
    driver_lookup = (
        drivers_df.drop_duplicates(subset=["driver_id"])
                  .set_index("driver_id").to_dict("index")
    )
    user_lookup = (
        users_df.drop_duplicates(subset=["user_id"])
                .set_index("user_id").to_dict("index")
    )
    device_lookup = (
        devices_df.set_index("device_id").to_dict("index")
    )

    driver_ids = drivers_df["driver_id"].unique().tolist()
    user_ids   = users_df["user_id"].unique().tolist()

    # Track velocity counters per driver and user
    # key: (id, date_str) → count
    driver_daily_rides  = {}
    user_weekly_rides   = {}
    driver_hourly_rides = {}

    # Track driver-user pair frequency for collusion detection
    pair_frequency = {}

    for i in range(N_TRANSACTIONS):
        txn_id = f"TXN{str(i+1).zfill(7)}"
        ts     = random_timestamp_2025()
        hour   = ts.hour
        month  = ts.month
        is_weekend = ts.weekday() >= 5
        is_holiday = ts.strftime("%Y-%m-%d") in INDIA_HOLIDAYS_2025

        # ── Assign driver and user ─────────────────────────────
        driver_id  = random.choice(driver_ids)
        user_id    = random.choice(user_ids)
        driver_row = driver_lookup.get(driver_id, {})
        user_row   = user_lookup.get(user_id, {})

        # ── Determine fraud type for this transaction ──────────
        rand_val = random.random()
        if rand_val < FRAUD_RATE_DRIVER:
            fraud_label = DRIVER_FRAUD
        elif rand_val < FRAUD_RATE_DRIVER + FRAUD_RATE_USER:
            fraud_label = USER_FRAUD
        elif rand_val < FRAUD_RATE_DRIVER + FRAUD_RATE_USER + FRAUD_RATE_COLLUSION:
            fraud_label = COLLUSION_FRAUD
        else:
            fraud_label = LEGIT

        is_fraud = int(fraud_label != LEGIT)

        # ── Zone and distance ──────────────────────────────────
        pickup_zone = random.choice(ZONE_NAMES)
        drop_zone   = random.choice(ZONE_NAMES)

        pickup_bounds = BANGALORE_ZONES[pickup_zone]
        drop_bounds   = BANGALORE_ZONES[drop_zone]

        pickup_lat = round(random.uniform(*pickup_bounds["lat"]), 6)
        pickup_lon = round(random.uniform(*pickup_bounds["lon"]), 6)
        drop_lat   = round(random.uniform(*drop_bounds["lat"]),   6)
        drop_lon   = round(random.uniform(*drop_bounds["lon"]),   6)

        claimed_distance_km = round(
            np.random.exponential(scale=6), 2
        )
        claimed_distance_km = float(np.clip(claimed_distance_km, 0.5, 45.0))

        # ── Vehicle and fare ───────────────────────────────────
        vehicle_type = driver_row.get("vehicle_type", "Mini")
        surge = 1.0
        if hour in range(8, 11) or hour in range(17, 21):
            surge += random.uniform(0.2, 0.7)
        if month in MONSOON_MONTHS:
            surge += random.uniform(0.1, 0.3)
        surge = round(np.clip(surge, 1.0, 4.0), 1)

        claimed_fare    = compute_expected_fare(
            vehicle_type, claimed_distance_km, surge
        )
        expected_fare   = compute_expected_fare(
            vehicle_type, claimed_distance_km, 1.0
        )

        # ── Payment ────────────────────────────────────────────
        payment_method = random.choices(PAYMENT_METHODS, PAYMENT_WEIGHTS)[0]

        # User fraud often uses card (stolen cards easier to abuse)
        if fraud_label == USER_FRAUD:
            payment_method = random.choices(
                ["Card", "UPI", "Wallet"],
                weights=[0.60, 0.25, 0.15]
            )[0]

        payment_status = "success"
        if fraud_label == USER_FRAUD and random.random() < 0.15:
            payment_status = "disputed"  # chargeback fraud
        elif random.random() < 0.03:
            payment_status = "failed"

        # ── Device signals ─────────────────────────────────────
        # Fraud transactions more likely to use risky devices
        if fraud_label in [USER_FRAUD, COLLUSION_FRAUD]:
            # Higher chance of high-risk device
            risky_devices = devices_df[
                devices_df["device_risk_level"] == "high"
            ]["device_id"].tolist()
            if risky_devices and random.random() < 0.40:
                user_device_id = random.choice(risky_devices)
            else:
                user_device_id = user_row.get("device_id",
                    f"DEV{str(random.randint(1, N_DEVICES)).zfill(6)}")
        else:
            user_device_id = user_row.get("device_id",
                f"DEV{str(random.randint(1, N_DEVICES)).zfill(6)}")

        device_row = device_lookup.get(user_device_id, {})
        uses_vpn          = device_row.get("uses_vpn", False)
        gps_mock_detected = device_row.get("gps_mock_detected", False)
        accounts_linked   = device_row.get("accounts_linked", 1)

        # GPS spoofing drivers always have mock GPS detected
        if fraud_label == DRIVER_FRAUD:
            gps_mock_detected = random.random() < 0.70

        # ── GPS and completion signals ─────────────────────────
        if fraud_label == LEGIT:
            signals = generate_legitimate_signals(
                claimed_distance_km, vehicle_type, surge, ts, hour
            )
        elif fraud_label == DRIVER_FRAUD:
            signals = generate_driver_fraud_signals(
                claimed_distance_km, vehicle_type, surge, ts, hour
            )
        elif fraud_label == USER_FRAUD:
            signals = generate_user_fraud_signals(
                claimed_distance_km, vehicle_type, surge, ts, hour
            )
        else:
            signals = generate_collusion_fraud_signals(
                claimed_distance_km, vehicle_type, surge, ts, hour
            )

        # ── Velocity features ──────────────────────────────────
        day_key  = (driver_id, ts.strftime("%Y-%m-%d"))
        hour_key = (driver_id, ts.strftime("%Y-%m-%d-%H"))
        week_key = (user_id, ts.strftime("%Y-W%W"))

        driver_rides_today  = driver_daily_rides.get(day_key, 0)
        driver_rides_this_hour = driver_hourly_rides.get(hour_key, 0)
        user_rides_this_week = user_weekly_rides.get(week_key, 0)

        # Fraud drivers artificially inflate their ride counts
        if fraud_label == DRIVER_FRAUD:
            driver_rides_today = max(
                driver_rides_today,
                random.randint(15, 30)
            )

        # Update counters
        driver_daily_rides[day_key]     = driver_rides_today + 1
        driver_hourly_rides[hour_key]   = driver_rides_this_hour + 1
        user_weekly_rides[week_key]     = user_rides_this_week + 1

        # ── Driver-User pair frequency ─────────────────────────
        pair_key = (driver_id, user_id)
        pair_freq = pair_frequency.get(pair_key, 0)

        # Collusion pairs ride together very frequently
        if fraud_label == COLLUSION_FRAUD:
            pair_freq = max(pair_freq, random.randint(8, 25))

        pair_frequency[pair_key] = pair_freq + 1

        # ── Account age signals ────────────────────────────────
        user_account_age  = user_row.get("account_age_days", 365)
        past_chargebacks  = user_row.get("past_chargebacks", 0)
        promo_abuse_count = user_row.get("promo_abuse_count", 0)
        payment_fail_rate = user_row.get("payment_failure_rate", 0.02)

        driver_account_age = max(1, int(
            driver_row.get("experience_years", 1) * 365
        ))
        past_fraud_flags = driver_row.get("past_fraud_flags", 0)
        gps_anomaly_count = driver_row.get("gps_anomaly_count", 0)

        # ── Promo usage ────────────────────────────────────────
        # User fraud often involves promo abuse
        promo_applied = False
        promo_code    = None
        if fraud_label == USER_FRAUD and random.random() < 0.55:
            promo_applied = True
            promo_code    = fake.bothify(text="PROMO##??")
        elif random.random() < 0.12:
            promo_applied = True
            promo_code    = fake.bothify(text="RIDE##??")

        # ── Fare anomaly score ─────────────────────────────────
        # How much does claimed fare deviate from expected fare?
        fare_anomaly_score = round(
            abs(claimed_fare - expected_fare) / max(expected_fare, 1), 3
        )

        # ── Intentional messiness ──────────────────────────────
        if random.random() < 0.02:
            claimed_distance_km = None
        if random.random() < 0.02:
            signals["gps_signal_strength"] = None
        if random.random() < 0.015:
            claimed_fare = None
        if random.random() < 0.03:
            pickup_zone = None

        records.append({
            # IDs
            "txn_id"                    : txn_id,
            "driver_id"                 : driver_id,
            "user_id"                   : user_id,
            "user_device_id"            : user_device_id,

            # Time
            "txn_timestamp"             : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour"                      : hour,
            "month"                     : month,
            "day_of_week"               : ts.strftime("%A"),
            "is_weekend"                : is_weekend,
            "is_holiday"                : is_holiday,

            # Geography
            "pickup_zone"               : pickup_zone,
            "drop_zone"                 : drop_zone,
            "pickup_lat"                : pickup_lat,
            "pickup_lon"                : pickup_lon,
            "drop_lat"                  : drop_lat,
            "drop_lon"                  : drop_lon,

            # Trip details
            "vehicle_type"              : vehicle_type,
            "claimed_distance_km"       : claimed_distance_km,
            "claimed_fare"              : claimed_fare,
            "expected_fare"             : expected_fare,
            "fare_anomaly_score"        : fare_anomaly_score,
            "surge_multiplier"          : surge,
            "payment_method"            : payment_method,
            "payment_status"            : payment_status,
            "promo_applied"             : promo_applied,
            "promo_code"                : promo_code,

            # GPS and completion signals
            "completion_time_min"       : signals["completion_time_min"],
            "distance_moved_km"         : signals["distance_moved_km"],
            "gps_signal_strength"       : signals["gps_signal_strength"],
            "gps_points_recorded"       : signals["gps_points_recorded"],
            "route_deviation_score"     : signals["route_deviation_score"],
            "claimed_speed_kmph"        : signals["claimed_speed_kmph"],

            # Device signals
            "uses_vpn"                  : uses_vpn,
            "gps_mock_detected"         : gps_mock_detected,
            "device_accounts_linked"    : accounts_linked,

            # Velocity features
            "driver_rides_today"        : driver_rides_today,
            "driver_rides_this_hour"    : driver_rides_this_hour,
            "user_rides_this_week"      : user_rides_this_week,

            # Relationship features
            "driver_user_pair_frequency": pair_freq,

            # Account signals
            "user_account_age_days"     : user_account_age,
            "user_past_chargebacks"     : past_chargebacks,
            "user_promo_abuse_count"    : promo_abuse_count,
            "user_payment_fail_rate"    : payment_fail_rate,
            "driver_account_age_days"   : driver_account_age,
            "driver_past_fraud_flags"   : past_fraud_flags,
            "driver_gps_anomaly_count"  : gps_anomaly_count,

            # TARGET VARIABLES
            "is_fraud"                  : is_fraud,        # Stage 1
            "fraud_label"               : fraud_label,     # Stage 2
        })

    df = pd.DataFrame(records)

    # ~0.5% duplicate transactions — system glitch
    dupes = df.sample(frac=0.005, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)
    df    = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  MAIN
# ================================================================

def main():
    print("=" * 65)
    print(" RealWorld-DS-ML | Ride Sharing | Fraud Detection")
    print(" City         : Bangalore")
    print(" Year         : 2025 (full year)")
    print(" Stage 1      : Binary — is_fraud (0/1)")
    print(" Stage 2      : Multi-class — fraud_label (0/1/2/3)")
    print(" Architecture : Hierarchical + Anomaly Detection")
    print("=" * 65)
    print(f" Transactions : {N_TRANSACTIONS:,}")
    print(f" Drivers      : {N_DRIVERS:,}")
    print(f" Users        : {N_USERS:,}")
    print(f" Devices      : {N_DEVICES:,}")
    print(f" Output dir   : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 65)

    devices_df     = generate_devices()
    drivers_df     = generate_drivers()
    users_df       = generate_users()
    transactions_df= generate_transactions(drivers_df, users_df, devices_df)

    devices_df.to_csv(     os.path.join(OUTPUT_DIR, "devices.csv"),      index=False)
    drivers_df.to_csv(     os.path.join(OUTPUT_DIR, "drivers.csv"),      index=False)
    users_df.to_csv(       os.path.join(OUTPUT_DIR, "users.csv"),        index=False)
    transactions_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)

    print()
    print("=" * 65)
    print(" ✅ All tables saved to data/raw/")
    print("=" * 65)
    print(f"  devices.csv      : {len(devices_df):>8,} rows")
    print(f"  drivers.csv      : {len(drivers_df):>8,} rows")
    print(f"  users.csv        : {len(users_df):>8,} rows")
    print(f"  transactions.csv : {len(transactions_df):>8,} rows")
    print("=" * 65)

    # ── Class distribution ────────────────────────────────────
    print()
    print(" Stage 1 — is_fraud distribution:")
    fraud_counts = transactions_df["is_fraud"].value_counts().sort_index()
    for k, v in fraud_counts.items():
        label = "Legitimate" if k == 0 else "Fraud"
        pct   = v / len(transactions_df) * 100
        print(f"   {label:<12}: {v:>7,}  ({pct:.2f}%)")

    print()
    print(" Stage 2 — fraud_label distribution:")
    label_map = {0:"Legitimate",1:"Driver Fraud",
                 2:"User Fraud",3:"Collusion Fraud"}
    label_counts = transactions_df["fraud_label"].value_counts().sort_index()
    for k, v in label_counts.items():
        pct = v / len(transactions_df) * 100
        print(f"   {label_map[k]:<16}: {v:>7,}  ({pct:.2f}%)")

    print("=" * 65)
    print()
    print(" Next step → Open notebooks/01_data_generation.ipynb")
    print("=" * 65)


if __name__ == "__main__":
    main()