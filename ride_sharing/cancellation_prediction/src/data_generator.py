"""
===============================================================
 RealWorld-DS-ML | Ride Sharing | Cancellation Prediction
 File    : data_generator.py
 Purpose : Generate realistic, production-style messy data
           simulating ride cancellation patterns on a
           Bangalore ride sharing platform (Ola/Uber/Rapido).

 Tables Generated:
   1. rides.csv    — core ride records with cancellation outcome
   2. drivers.csv  — driver profiles with behavioral history
   3. users.csv    — user profiles with cancellation history
   4. weather.csv  — hourly weather per zone (Bangalore 2025)

 Target Variable:
   ride_outcome:
     0 = completed
     1 = cancelled_by_driver
     2 = cancelled_by_user

 Class Distribution (realistic):
   Completed         : ~72%
   Cancelled by user : ~18%
   Cancelled by driver: ~10%

 Key Design Decisions:
   - Driver cancellations driven by: long pickup distance,
     high daily cancellations already, low user rating,
     peak hour with better rides available nearby
   - User cancellations driven by: long wait time, high surge,
     indecisive user history, wrong pickup location,
     late night discomfort
   - Intentional messiness: nulls, duplicates, outliers,
     inconsistent formats — exactly like production data
===============================================================
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
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
N_RIDES   = 60_000
N_DRIVERS = 3_000
N_USERS   = 20_000

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

# ── Zone Cancellation Risk ────────────────────────────────────
# Zones with narrow lanes, poor GPS accuracy, or heavy traffic
# have higher baseline cancellation rates
ZONE_CANCELLATION_RISK = {
    "Koramangala"       : 0.72,
    "Whitefield"        : 0.65,
    "Indiranagar"       : 0.70,
    "HSR Layout"        : 0.68,
    "Electronic City"   : 0.60,
    "Marathahalli"      : 0.67,
    "Bellandur"         : 0.66,
    "Sarjapur Road"     : 0.63,
    "BTM Layout"        : 0.71,
    "Jayanagar"         : 0.58,
    "JP Nagar"          : 0.55,
    "Banashankari"      : 0.54,
    "Rajajinagar"       : 0.60,
    "Malleswaram"       : 0.62,
    "Hebbal"            : 0.64,
    "Yelahanka"         : 0.50,
    "Yeshwanthpur"      : 0.61,
    "Cunningham Road"   : 0.69,
    "MG Road"           : 0.75,   # highest — commercial congestion
    "Shivajinagar"      : 0.68,
    "KR Puram"          : 0.59,
    "Bannerghatta Road" : 0.52,
    "Kadugodi"          : 0.55,
    "Nagawara"          : 0.57,
    "Vijayanagar"       : 0.53,
}

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

# ── Constants ────────────────────────────────────────────────
VEHICLE_TYPES   = ["Auto",  "Mini",   "Sedan",  "SUV",    "Bike"]
VEHICLE_WEIGHTS = [0.20,     0.30,     0.25,     0.15,     0.10]

MONSOON_MONTHS  = {6, 7, 8, 9}

CANCELLATION_REASONS_DRIVER = [
    "Pickup location too far",
    "Found better ride nearby",
    "Vehicle breakdown",
    "Personal emergency",
    "User unresponsive",
    "Low user rating",
    None,   # missing reason — realistic
]

CANCELLATION_REASONS_USER = [
    "Wait time too long",
    "Found alternate transport",
    "Wrong pickup location",
    "Surge price too high",
    "Personal reason",
    "App issue",
    "Driver not moving",
    None,
]

# ── Outcome Labels ───────────────────────────────────────────
OUTCOME_COMPLETED        = 0
OUTCOME_CANCELLED_DRIVER = 1
OUTCOME_CANCELLED_USER   = 2


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


def get_weather_weights(month):
    """Weather condition probability weights by month."""
    if month in MONSOON_MONTHS:
        return [0.20, 0.25, 0.25, 0.18, 0.07, 0.05]
    elif month in {11, 12, 1, 2}:
        return [0.55, 0.25, 0.08, 0.03, 0.01, 0.08]
    else:
        return [0.45, 0.28, 0.14, 0.07, 0.03, 0.03]


def compute_estimated_wait_time(driver_distance_km, hour, weather_condition):
    """
    Estimate wait time in minutes based on:
    - How far the driver is
    - Traffic (peak hours take longer)
    - Weather (rain slows everything down)
    """
    # Base: ~3 minutes per km in Bangalore traffic
    base_wait = driver_distance_km * 3.0

    # Peak hour traffic multiplier
    if hour in range(8, 11) or hour in range(17, 21):
        base_wait *= random.uniform(1.4, 1.8)
    elif hour in range(11, 17):
        base_wait *= random.uniform(1.1, 1.3)

    # Rain slows traffic
    if weather_condition in ["Heavy Rain", "Storm"]:
        base_wait *= random.uniform(1.5, 2.0)
    elif weather_condition == "Light Rain":
        base_wait *= random.uniform(1.1, 1.4)

    return round(base_wait + random.uniform(-1, 2), 1)


def compute_surge_at_booking(hour, is_weekend, is_holiday, weather_condition):
    """
    Simplified surge at booking time.
    Users who see high surge are more likely to cancel.
    """
    surge = 1.0
    if hour in range(8, 11) or hour in range(17, 21):
        surge += random.uniform(0.3, 0.8)
    if is_weekend:
        surge += random.uniform(0.1, 0.3)
    if is_holiday:
        surge += random.uniform(0.3, 0.6)
    if weather_condition in ["Heavy Rain", "Storm"]:
        surge += random.uniform(0.3, 0.7)
    elif weather_condition == "Light Rain":
        surge += random.uniform(0.1, 0.3)
    surge += random.uniform(-0.1, 0.15)
    return round(np.clip(surge, 1.0, 4.5), 1)


def decide_ride_outcome(
    # Driver factors
    driver_distance_km,
    driver_cancellations_today,
    driver_rating,
    driver_acceptance_rate,
    # User factors
    user_rating,
    user_cancellations_last_30d,
    user_booking_attempts,
    # Contextual factors
    estimated_wait_time,
    surge_at_booking,
    hour,
    is_weekend,
    is_holiday,
    weather_condition,
    pickup_zone,
):
    """
    Core business logic: decide the ride outcome.
    This is the TARGET VARIABLE generation function.

    Simulates the decision-making process of both driver
    and user independently, then determines the final outcome.

    Returns:
        outcome     : 0 (completed), 1 (driver cancel), 2 (user cancel)
        cancelled_by: 'driver', 'user', or None
        reason      : cancellation reason string or None
        time_to_cancel: minutes after booking (None if completed)
    """

    # ── DRIVER CANCELLATION PROBABILITY ──────────────────────
    # Each factor independently contributes to the driver's
    # likelihood of cancelling

    driver_cancel_prob = 0.08   # base rate ~8%

    # Long pickup distance — primary driver for driver cancellation
    if driver_distance_km > 5:
        driver_cancel_prob += 0.15
    elif driver_distance_km > 3:
        driver_cancel_prob += 0.08
    elif driver_distance_km > 1.5:
        driver_cancel_prob += 0.03

    # Already cancelled multiple times today — habit/burnout
    if driver_cancellations_today >= 4:
        driver_cancel_prob += 0.20
    elif driver_cancellations_today >= 2:
        driver_cancel_prob += 0.10
    elif driver_cancellations_today == 1:
        driver_cancel_prob += 0.04

    # Low acceptance rate driver — generally unreliable
    if driver_acceptance_rate is not None:
        if driver_acceptance_rate < 0.6:
            driver_cancel_prob += 0.12
        elif driver_acceptance_rate < 0.75:
            driver_cancel_prob += 0.05

    # Low user rating — driver cherry picks good users
    if user_rating is not None:
        if user_rating < 3.5:
            driver_cancel_prob += 0.15
        elif user_rating < 4.0:
            driver_cancel_prob += 0.06

    # Peak hour — better rides available nearby
    if hour in range(8, 11) or hour in range(17, 21):
        driver_cancel_prob += 0.06

    # Zone risk — MG Road, BTM Layout have higher cancellation
    zone_risk = ZONE_CANCELLATION_RISK.get(pickup_zone, 0.6)
    driver_cancel_prob += (zone_risk - 0.6) * 0.2

    # Random noise
    driver_cancel_prob += random.uniform(-0.02, 0.03)
    driver_cancel_prob  = np.clip(driver_cancel_prob, 0, 0.70)

    # ── USER CANCELLATION PROBABILITY ────────────────────────
    user_cancel_prob = 0.12   # base rate ~12%

    # Long wait time — most common user cancellation reason
    if estimated_wait_time > 15:
        user_cancel_prob += 0.25
    elif estimated_wait_time > 10:
        user_cancel_prob += 0.15
    elif estimated_wait_time > 7:
        user_cancel_prob += 0.07

    # High surge — price shock after booking
    if surge_at_booking > 2.5:
        user_cancel_prob += 0.20
    elif surge_at_booking > 2.0:
        user_cancel_prob += 0.12
    elif surge_at_booking > 1.5:
        user_cancel_prob += 0.05

    # Habitual canceller — user's historical behavior
    if user_cancellations_last_30d >= 8:
        user_cancel_prob += 0.22
    elif user_cancellations_last_30d >= 4:
        user_cancel_prob += 0.12
    elif user_cancellations_last_30d >= 2:
        user_cancel_prob += 0.05

    # Indecisive user — opened app many times before booking
    if user_booking_attempts >= 4:
        user_cancel_prob += 0.15
    elif user_booking_attempts >= 2:
        user_cancel_prob += 0.07

    # Rain — user may find auto/walk better
    if weather_condition in ["Heavy Rain", "Storm"]:
        user_cancel_prob += 0.08
    elif weather_condition == "Light Rain":
        user_cancel_prob += 0.04

    # Late night — some users cancel out of safety concerns
    if hour in range(23, 24) or hour in range(0, 3):
        user_cancel_prob += 0.05

    # Holiday — users more likely to find alternate plans
    if is_holiday:
        user_cancel_prob += 0.06

    # Random noise
    user_cancel_prob += random.uniform(-0.02, 0.03)
    user_cancel_prob  = np.clip(user_cancel_prob, 0, 0.75)

    # ── DECIDE FINAL OUTCOME ──────────────────────────────────
    # Both driver and user make independent decisions
    # Driver gets first chance to cancel (happens first in real flow)
    driver_cancels = random.random() < driver_cancel_prob
    user_cancels   = random.random() < user_cancel_prob

    if driver_cancels and user_cancels:
        # Both wanted to cancel — whoever acts first wins
        # Driver slightly more likely to cancel first
        if random.random() < 0.6:
            outcome      = OUTCOME_CANCELLED_DRIVER
            cancelled_by = "driver"
            reason       = random.choice(CANCELLATION_REASONS_DRIVER)
            time_to_cancel = round(random.uniform(0.5, 5.0), 1)
        else:
            outcome      = OUTCOME_CANCELLED_USER
            cancelled_by = "user"
            reason       = random.choice(CANCELLATION_REASONS_USER)
            time_to_cancel = round(random.uniform(0.5, 8.0), 1)

    elif driver_cancels:
        outcome      = OUTCOME_CANCELLED_DRIVER
        cancelled_by = "driver"
        reason       = random.choice(CANCELLATION_REASONS_DRIVER)
        time_to_cancel = round(random.uniform(0.5, 5.0), 1)

    elif user_cancels:
        outcome      = OUTCOME_CANCELLED_USER
        cancelled_by = "user"
        reason       = random.choice(CANCELLATION_REASONS_USER)
        time_to_cancel = round(random.uniform(1.0, 12.0), 1)

    else:
        outcome        = OUTCOME_COMPLETED
        cancelled_by   = None
        reason         = None
        time_to_cancel = None

    return outcome, cancelled_by, reason, time_to_cancel


# ================================================================
#  TABLE 1 — USERS
# ================================================================

def generate_users():
    print("Generating users table...")
    records = []

    for i in range(N_USERS):
        user_id = f"USR{str(i+1).zfill(6)}"
        signup  = datetime(2025, 1, 1) - timedelta(
            days=random.randint(0, 1460)
        )

        # Cancellation behavior — key signal for user cancel prediction
        # Most users have low cancellation history
        # Small % are habitual cancellers
        cancel_profile = random.choices(
            ["low", "moderate", "high"],
            weights=[0.65, 0.25, 0.10]
        )[0]

        if cancel_profile == "low":
            cancellations_last_30d = random.randint(0, 1)
            total_cancellations    = random.randint(0, 5)
        elif cancel_profile == "moderate":
            cancellations_last_30d = random.randint(2, 4)
            total_cancellations    = random.randint(5, 20)
        else:
            cancellations_last_30d = random.randint(5, 12)
            total_cancellations    = random.randint(20, 80)

        total_rides = random.randint(
            max(1, total_cancellations), 300
        )

        rating = round(np.random.normal(4.2, 0.5), 1)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.04:
            rating = None

        age = random.randint(18, 60)
        if random.random() < 0.02:
            age = random.choice([-1, 0, 150, 999])

        records.append({
            "user_id"                   : user_id,
            "name"                      : fake.name(),
            "phone"                     : fake.phone_number() if random.random() > 0.03 else None,
            "email"                     : fake.email() if random.random() > 0.05 else None,
            "age"                       : age,
            "gender"                    : random.choice(["Male","Female","Other",None]),
            "home_zone"                 : random.choice(ZONE_NAMES),
            "work_zone"                 : random.choice(ZONE_NAMES),
            "signup_date"               : signup.strftime("%Y-%m-%d"),
            "rating"                    : rating,
            "total_rides"               : total_rides,
            "total_cancellations"       : total_cancellations,
            "cancellations_last_30d"    : cancellations_last_30d,
            "cancel_profile"            : cancel_profile,
            "preferred_vehicle"         : random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0],
            "is_prime_member"           : random.choice([True, False, None]),
            "avg_booking_attempts"      : round(random.uniform(1.0, 4.0), 1),
        })

    df = pd.DataFrame(records)

    # ~1% duplicates
    dupes = df.sample(frac=0.01, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)

    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
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

        vehicle_type = random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0]

        # Driver reliability profile — key signal for driver cancel prediction
        reliability = random.choices(
            ["reliable", "average", "unreliable"],
            weights=[0.55, 0.35, 0.10]
        )[0]

        if reliability == "reliable":
            acceptance_rate    = round(random.uniform(0.82, 1.00), 2)
            cancellation_rate  = round(random.uniform(0.00, 0.08), 2)
            rating             = round(np.random.normal(4.5, 0.3), 2)
        elif reliability == "average":
            acceptance_rate    = round(random.uniform(0.65, 0.82), 2)
            cancellation_rate  = round(random.uniform(0.08, 0.18), 2)
            rating             = round(np.random.normal(4.1, 0.4), 2)
        else:
            acceptance_rate    = round(random.uniform(0.40, 0.65), 2)
            cancellation_rate  = round(random.uniform(0.18, 0.40), 2)
            rating             = round(np.random.normal(3.5, 0.5), 2)

        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.03:
            rating = None
        if random.random() < 0.05:
            acceptance_rate = None

        exp_years = round(
            (datetime(2025, 12, 31) - joined_date).days / 365, 1
        )

        vehicle_model = random.choice([
            "Maruti Swift", "Hyundai i20", "Honda City",
            "Toyota Etios", "Maruti Dzire", "Hyundai Xcent",
            "Bajaj RE Auto", "TVS Jupiter", "Honda Activa",
            "Toyota Innova", None
        ])

        records.append({
            "driver_id"              : driver_id,
            "name"                   : fake.name(),
            "phone"                  : fake.phone_number() if random.random() > 0.02 else None,
            "home_zone"              : random.choice(ZONE_NAMES),
            "vehicle_type"           : vehicle_type,
            "vehicle_model"          : vehicle_model,
            "vehicle_year"           : random.choice([2016,2017,2018,2019,2020,2021,2022,2023,None]),
            "rating"                 : rating,
            "reliability_profile"    : reliability,
            "total_rides"            : random.randint(50, 10000),
            "acceptance_rate"        : acceptance_rate,
            "cancellation_rate"      : cancellation_rate,
            "joined_date"            : joined_date.strftime("%Y-%m-%d"),
            "experience_years"       : exp_years,
            "is_active"              : random.choice([True, False]),
            "online_hours_per_day"   : round(random.uniform(2, 14), 1) if random.random() > 0.04 else None,
            "preferred_zone"         : random.choice(ZONE_NAMES),
        })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  TABLE 3 — WEATHER
# ================================================================

def generate_weather():
    """Hourly weather for 10 key Bangalore zones — full year 2025."""
    print("Generating weather table...")
    records  = []
    conditions = ["Clear","Cloudy","Light Rain","Heavy Rain","Storm","Foggy"]
    key_zones  = [
        "Koramangala","Whitefield","Electronic City","Marathahalli",
        "MG Road","Hebbal","HSR Layout","Indiranagar","JP Nagar","Yelahanka"
    ]
    start = datetime(2025, 1, 1, 0, 0)
    hours = 365 * 24

    for zone in key_zones:
        for h in range(hours):
            ts      = start + timedelta(hours=h)
            month   = ts.month
            weights = get_weather_weights(month)
            condition = random.choices(conditions, weights)[0]

            if month in {12, 1, 2}:
                temp = round(np.random.normal(20, 3), 1)
            elif month in {3, 4, 5}:
                temp = round(np.random.normal(33, 4), 1)
            elif month in MONSOON_MONTHS:
                temp = round(np.random.normal(25, 3), 1)
            else:
                temp = round(np.random.normal(27, 3), 1)

            humidity   = round(random.uniform(40, 95), 1)
            wind_speed = round(random.uniform(0, 35), 1)

            if random.random() < 0.02:
                temp = None
            if random.random() < 0.02:
                humidity = None

            records.append({
                "zone"       : zone,
                "timestamp"  : ts.strftime("%Y-%m-%d %H:00:00"),
                "month"      : month,
                "condition"  : condition,
                "temperature": temp,
                "humidity"   : humidity,
                "wind_speed" : wind_speed,
            })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  TABLE 4 — RIDES  (main fact table)
# ================================================================

def generate_rides(drivers_df, users_df, weather_df):
    print("Generating rides table... this may take a moment")
    records = []

    driver_ids = drivers_df["driver_id"].tolist()
    user_ids   = users_df["user_id"].tolist()

    # Build lookup dicts for fast driver/user attribute access
    driver_lookup = drivers_df.drop_duplicates(subset="driver_id", keep="first").set_index("driver_id").to_dict("index")
    user_lookup   = users_df.drop_duplicates(subset="user_id", keep="first").set_index("user_id").to_dict("index")

    # Build weather index for fast lookup
    weather_index = {}
    weather_zones = weather_df["zone"].unique().tolist()
    for _, row in weather_df.iterrows():
        key = (row["zone"], row["timestamp"])
        weather_index[key] = row["condition"]

    # Track driver cancellations per day — realistic daily counter
    driver_daily_cancels = {}

    for i in range(N_RIDES):
        ride_id      = f"RID{str(i+1).zfill(7)}"
        ts           = random_timestamp_2025()
        hour         = ts.hour
        month        = ts.month
        is_weekend   = ts.weekday() >= 5
        is_holiday   = ts.strftime("%Y-%m-%d") in INDIA_HOLIDAYS_2025

        pickup_zone  = random.choice(ZONE_NAMES)
        drop_zone    = random.choice(ZONE_NAMES)

        pickup_bounds = BANGALORE_ZONES[pickup_zone]
        drop_bounds   = BANGALORE_ZONES[drop_zone]
        pickup_lat    = round(random.uniform(*pickup_bounds["lat"]), 6)
        pickup_lon    = round(random.uniform(*pickup_bounds["lon"]), 6)
        drop_lat      = round(random.uniform(*drop_bounds["lat"]), 6)
        drop_lon      = round(random.uniform(*drop_bounds["lon"]), 6)

        vehicle_type  = random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0]

        # ── Assign driver ─────────────────────────────────────
        driver_id  = random.choice(driver_ids)
        driver_row = driver_lookup.get(driver_id, {})

        driver_rating          = driver_row.get("rating", 4.0)
        driver_acceptance_rate = driver_row.get("acceptance_rate", 0.75)

        # Driver distance to pickup — exponential distribution
        # Most drivers are nearby, some are far
        driver_distance_km = round(np.random.exponential(scale=2.5), 2)
        driver_distance_km = float(np.clip(driver_distance_km, 0.2, 15.0))

        # Driver cancellations today
        day_key = (driver_id, ts.strftime("%Y-%m-%d"))
        driver_cancellations_today = driver_daily_cancels.get(day_key, 0)

        # ── Assign user ───────────────────────────────────────
        user_id  = random.choice(user_ids)
        user_row = user_lookup.get(user_id, {})

        user_rating                 = user_row.get("rating", 4.0)
        user_cancellations_last_30d = user_row.get("cancellations_last_30d", 0)
        avg_booking_attempts        = user_row.get("avg_booking_attempts", 1.5)

        # This ride's booking attempts — varies around user's average
        user_booking_attempts = max(1, int(np.random.normal(
            avg_booking_attempts, 0.8
        )))

        # ── Weather ───────────────────────────────────────────
        weather_hour = ts.strftime("%Y-%m-%d %H:00:00")
        lookup_zone  = pickup_zone if pickup_zone in weather_zones \
                       else random.choice(weather_zones)
        weather_cond = weather_index.get((lookup_zone, weather_hour), "Clear")

        # ── Derived features ──────────────────────────────────
        estimated_wait_time = compute_estimated_wait_time(
            driver_distance_km, hour, weather_cond
        )
        surge_at_booking = compute_surge_at_booking(
            hour, is_weekend, is_holiday, weather_cond
        )

        # Pickup accuracy — how precisely user dropped the pin
        # Low accuracy = driver struggles to find user = more cancellations
        pickup_accuracy = round(random.uniform(0.3, 1.0), 2)
        if random.random() < 0.03:
            pickup_accuracy = None

        # ── Decide outcome ────────────────────────────────────
        outcome, cancelled_by, cancel_reason, time_to_cancel = decide_ride_outcome(
            driver_distance_km       = driver_distance_km,
            driver_cancellations_today = driver_cancellations_today,
            driver_rating            = driver_rating,
            driver_acceptance_rate   = driver_acceptance_rate,
            user_rating              = user_rating,
            user_cancellations_last_30d = user_cancellations_last_30d,
            user_booking_attempts    = user_booking_attempts,
            estimated_wait_time      = estimated_wait_time,
            surge_at_booking         = surge_at_booking,
            hour                     = hour,
            is_weekend               = is_weekend,
            is_holiday               = is_holiday,
            weather_condition        = weather_cond,
            pickup_zone              = pickup_zone,
        )

        # Update driver daily cancel counter
        if outcome == OUTCOME_CANCELLED_DRIVER:
            driver_daily_cancels[day_key] = driver_cancellations_today + 1

        # Trip details — only for completed rides
        distance_km  = round(np.random.exponential(scale=6), 2)
        distance_km  = float(np.clip(distance_km, 0.5, 45.0))
        if outcome == OUTCOME_COMPLETED:
            duration_min = round(distance_km * random.uniform(3.0, 6.0), 1)
            fare_amount  = round(
                (30 + 12 * distance_km) * surge_at_booking, 2
            )
        else:
            duration_min = None
            fare_amount  = None
            distance_km  = None

        # Intentional messiness
        if random.random() < 0.03:
            pickup_zone = None
        if random.random() < 0.02:
            drop_zone = None
        if random.random() < 0.02:
            estimated_wait_time = None
        if random.random() < 0.015:
            driver_distance_km = None

        records.append({
            # IDs
            "ride_id"                    : ride_id,
            "driver_id"                  : driver_id,
            "user_id"                    : user_id,

            # Location
            "pickup_zone"                : pickup_zone,
            "drop_zone"                  : drop_zone,
            "pickup_lat"                 : pickup_lat,
            "pickup_lon"                 : pickup_lon,
            "drop_lat"                   : drop_lat,
            "drop_lon"                   : drop_lon,
            "pickup_accuracy_score"      : pickup_accuracy,

            # Time
            "ride_timestamp"             : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour"                       : hour,
            "month"                      : month,
            "day_of_week"                : ts.strftime("%A"),
            "is_weekend"                 : is_weekend,
            "is_holiday"                 : is_holiday,

            # Vehicle
            "vehicle_type"               : vehicle_type,

            # Weather
            "weather_condition"          : weather_cond,

            # Driver signals at time of booking
            "driver_distance_to_pickup_km" : driver_distance_km,
            "driver_rating_at_booking"   : driver_rating,
            "driver_acceptance_rate"     : driver_acceptance_rate,
            "driver_cancellations_today" : driver_cancellations_today,

            # User signals at time of booking
            "user_rating_at_booking"     : user_rating,
            "user_cancellations_last_30d": user_cancellations_last_30d,
            "user_booking_attempts"      : user_booking_attempts,

            # Contextual
            "estimated_wait_time_min"    : estimated_wait_time,
            "surge_at_booking"           : surge_at_booking,

            # Trip details (only for completed)
            "distance_km"                : distance_km,
            "duration_min"               : duration_min,
            "fare_amount"                : fare_amount,

            # Cancellation details
            "time_to_cancellation_min"   : time_to_cancel,
            "cancelled_by"               : cancelled_by,
            "cancellation_reason"        : cancel_reason,

            # TARGET VARIABLE
            "ride_outcome"               : outcome,
        })

    df = pd.DataFrame(records)

    # ~0.5% duplicate rows — system glitch
    dupes = df.sample(frac=0.005, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)
    df    = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  MAIN
# ================================================================

def main():
    print("=" * 62)
    print(" RealWorld-DS-ML | Ride Sharing | Cancellation Prediction")
    print(" City       : Bangalore")
    print(" Year       : 2025 (full year)")
    print(" Zones      : 25 real Bangalore zones")
    print(" Target     : ride_outcome (0=completed, 1=driver cancel,")
    print("                            2=user cancel)")
    print("=" * 62)
    print(f" Rides      : {N_RIDES:,}")
    print(f" Drivers    : {N_DRIVERS:,}")
    print(f" Users      : {N_USERS:,}")
    print(f" Output dir : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 62)

    users_df   = generate_users()
    drivers_df = generate_drivers()
    weather_df = generate_weather()
    rides_df   = generate_rides(drivers_df, users_df, weather_df)

    users_df.to_csv(   os.path.join(OUTPUT_DIR, "users.csv"),    index=False)
    drivers_df.to_csv( os.path.join(OUTPUT_DIR, "drivers.csv"),  index=False)
    weather_df.to_csv( os.path.join(OUTPUT_DIR, "weather.csv"),  index=False)
    rides_df.to_csv(   os.path.join(OUTPUT_DIR, "rides.csv"),    index=False)

    print()
    print("=" * 62)
    print(" ✅ All tables saved to data/raw/")
    print("=" * 62)
    print(f"  users.csv    : {len(users_df):>8,} rows")
    print(f"  drivers.csv  : {len(drivers_df):>8,} rows")
    print(f"  weather.csv  : {len(weather_df):>8,} rows")
    print(f"  rides.csv    : {len(rides_df):>8,} rows")
    print("=" * 62)

    # ── Class distribution check ──────────────────────────────
    print()
    print(" Target Variable Distribution:")
    counts = rides_df["ride_outcome"].value_counts().sort_index()
    labels = {0: "Completed        ", 1: "Cancelled-Driver ", 2: "Cancelled-User   "}
    for k, v in counts.items():
        pct = v / len(rides_df) * 100
        print(f"   {labels[k]}: {v:>7,}  ({pct:.1f}%)")
    print("=" * 62)
    print()
    print(" Next step → Open notebooks/01_data_generation.ipynb")
    print("=" * 62)


if __name__ == "__main__":
    main()