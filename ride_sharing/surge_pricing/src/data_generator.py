"""
===============================================================
 RealWorld-DS-ML | Ride Sharing | Surge Pricing
 File    : data_generator.py
 Purpose : Generate realistic, production-style messy data
           simulating an Ola/Uber/Rapido ride sharing platform
           specifically for Bangalore, India — full year 2025.

 Tables Generated:
   1. rides.csv         — core ride transaction records
   2. drivers.csv       — driver profiles and stats
   3. users.csv         — user/customer profiles
   4. payments.csv      — payment records per ride
   5. weather.csv       — weather conditions per hour per zone

 Design Philosophy:
   - Single city (Bangalore) — deep and realistic, not shallow
     and generic across multiple cities.
   - 25 real Bangalore zones with realistic lat/lon bounds.
   - Full year 2025 data — captures all seasonal patterns:
     monsoon surge, Diwali, New Year, summer etc.
   - Data is intentionally messy — nulls, duplicates, outliers,
     inconsistent formats — exactly like real production data.
   - Tables are joinable via ride_id, driver_id, user_id.
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
N_RIDES   = 50_000
N_DRIVERS = 3_000
N_USERS   = 20_000

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Bangalore Zones ──────────────────────────────────────────
# 25 real Bangalore zones with realistic lat/lon bounding boxes
# Each zone has its own lat/lon range for precise coordinate generation

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

# ── Zone Demand Profile ───────────────────────────────────────
# Reflects real Bangalore demand patterns
# Tech corridors naturally have higher demand during peak hours
ZONE_BASE_DEMAND = {
    "Koramangala"       : 0.85,
    "Whitefield"        : 0.90,
    "Indiranagar"       : 0.80,
    "HSR Layout"        : 0.82,
    "Electronic City"   : 0.88,
    "Marathahalli"      : 0.87,
    "Bellandur"         : 0.83,
    "Sarjapur Road"     : 0.78,
    "BTM Layout"        : 0.75,
    "Jayanagar"         : 0.65,
    "JP Nagar"          : 0.62,
    "Banashankari"      : 0.58,
    "Rajajinagar"       : 0.60,
    "Malleswaram"       : 0.63,
    "Hebbal"            : 0.70,
    "Yelahanka"         : 0.55,
    "Yeshwanthpur"      : 0.68,
    "Cunningham Road"   : 0.72,
    "MG Road"           : 0.88,
    "Shivajinagar"      : 0.74,
    "KR Puram"          : 0.65,
    "Bannerghatta Road" : 0.60,
    "Kadugodi"          : 0.58,
    "Nagawara"          : 0.62,
    "Vijayanagar"       : 0.57,
}

# ── India Public Holidays 2025 ────────────────────────────────
# Covers central government + major regional holidays
INDIA_HOLIDAYS_2025 = {
    "2025-01-01",   # New Year's Day
    "2025-01-14",   # Makar Sankranti / Pongal
    "2025-01-26",   # Republic Day
    "2025-02-26",   # Maha Shivratri
    "2025-03-14",   # Holi
    "2025-03-31",   # Id-ul-Fitr (Eid)
    "2025-04-06",   # Ram Navami
    "2025-04-10",   # Mahavir Jayanti
    "2025-04-14",   # Dr. Ambedkar Jayanti / Tamil New Year
    "2025-04-18",   # Good Friday
    "2025-05-12",   # Buddha Purnima
    "2025-06-07",   # Id-ul-Zuha (Bakrid)
    "2025-07-06",   # Muharram
    "2025-08-15",   # Independence Day
    "2025-08-16",   # Janmashtami
    "2025-09-05",   # Milad-un-Nabi
    "2025-10-02",   # Gandhi Jayanti
    "2025-10-02",   # Dussehra
    "2025-10-20",   # Diwali (Lakshmi Puja)
    "2025-10-21",   # Diwali (second day)
    "2025-11-05",   # Guru Nanak Jayanti
    "2025-11-01",   # Kannada Rajyotsava (Karnataka specific)
    "2025-12-25",   # Christmas
}

# ── Other Constants ───────────────────────────────────────────
VEHICLE_TYPES   = ["Auto",  "Mini",   "Sedan",  "SUV",    "Bike"]
VEHICLE_WEIGHTS = [0.20,     0.30,     0.25,     0.15,     0.10]

PAYMENT_METHODS = ["UPI", "Cash", "Card", "Wallet"]
PAY_WEIGHTS     = [0.50,  0.25,   0.15,   0.10]      # UPI dominant in Bangalore

CANCELLATION_REASONS = [
    "Driver took too long",
    "Found alternate transport",
    "Wrong pickup location",
    "Driver asked to cancel",
    "Personal reason",
    "App issue",
    None,
]

# Bangalore monsoon months — affects weather weights
MONSOON_MONTHS = {6, 7, 8, 9}   # June to September


# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def random_timestamp_2025():
    """Return a random datetime within full year 2025."""
    start = datetime(2025, 1, 1, 0, 0, 0)
    end   = datetime(2025, 12, 31, 23, 59, 59)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def get_weather_weights(month):
    """
    Adjust weather condition probabilities by month.
    Bangalore monsoon (Jun-Sep) has much higher rain probability.
    """
    if month in MONSOON_MONTHS:
        return [0.20, 0.25, 0.25, 0.18, 0.07, 0.05]   # more rain
    elif month in {11, 12, 1, 2}:
        return [0.55, 0.25, 0.08, 0.03, 0.01, 0.08]   # mostly clear, some fog in winter
    else:
        return [0.45, 0.28, 0.14, 0.07, 0.03, 0.03]   # normal months


def compute_surge_multiplier(hour, is_weekend, is_holiday,
                              weather_condition, zone, month):
    """
    Core business logic: compute surge multiplier.
    This is the TARGET VARIABLE we will predict.

    Factors considered:
      1. Time of day — morning/evening peak hours
      2. Weekend vs weekday
      3. Public holiday
      4. Weather condition — rain drives surge in Bangalore
      5. Zone base demand — tech corridors surge more
      6. Month — accounts for seasonal patterns
    """
    surge = 1.0

    # ── Time of day ──────────────────────────────────────────
    if hour in range(8, 11):
        surge += np.random.uniform(0.4, 0.9)    # morning office rush
    elif hour in range(17, 21):
        surge += np.random.uniform(0.4, 0.9)    # evening office rush
    elif hour in range(13, 15):
        surge += np.random.uniform(0.1, 0.3)    # mild lunch surge
    elif hour in range(23, 24) or hour in range(0, 3):
        surge += np.random.uniform(0.2, 0.5)    # late night / pub hours

    # ── Weekend ──────────────────────────────────────────────
    if is_weekend:
        if hour in range(10, 14) or hour in range(19, 23):
            surge += np.random.uniform(0.2, 0.4)   # weekend outing hours
        else:
            surge += np.random.uniform(0.05, 0.2)

    # ── Holiday ──────────────────────────────────────────────
    if is_holiday:
        surge += np.random.uniform(0.3, 0.7)

    # ── Weather — Bangalore rain causes significant surge ────
    if weather_condition == "Storm":
        surge += np.random.uniform(0.5, 1.0)
    elif weather_condition == "Heavy Rain":
        surge += np.random.uniform(0.3, 0.6)
    elif weather_condition == "Light Rain":
        surge += np.random.uniform(0.1, 0.3)
    elif weather_condition == "Foggy":
        surge += np.random.uniform(0.1, 0.2)

    # ── Zone demand ──────────────────────────────────────────
    base_demand = ZONE_BASE_DEMAND.get(zone, 0.6)
    surge += base_demand * np.random.uniform(0.1, 0.4)

    # ── Seasonal month effect ────────────────────────────────
    if month in MONSOON_MONTHS:
        surge += np.random.uniform(0.05, 0.2)   # monsoon base lift
    elif month == 12:
        surge += np.random.uniform(0.1, 0.3)    # December festive/year-end

    # ── Random noise ─────────────────────────────────────────
    surge += np.random.uniform(-0.1, 0.15)

    # ── Cap between 1.0 and 4.5 ──────────────────────────────
    surge = round(np.clip(surge, 1.0, 4.5), 1)
    return surge


# ================================================================
#  TABLE 1 — USERS
# ================================================================

def generate_users():
    print("Generating users table...")
    records = []

    for i in range(N_USERS):
        user_id = f"USR{str(i+1).zfill(6)}"
        signup  = datetime(2025, 1, 1) - timedelta(days=random.randint(0, 1460))  # up to 4 yrs ago

        phone = fake.phone_number() if random.random() > 0.03 else None   # 3% missing
        email = fake.email()        if random.random() > 0.05 else None   # 5% missing
        age   = random.randint(18, 60)
        if random.random() < 0.02:
            age = random.choice([-1, 0, 150, 999])                        # 2% garbage

        rating = round(np.random.normal(4.2, 0.5), 1)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.04:
            rating = None                                                  # 4% missing

        home_zone = random.choice(ZONE_NAMES)
        work_zone = random.choice(ZONE_NAMES)

        records.append({
            "user_id"          : user_id,
            "name"             : fake.name(),
            "phone"            : phone,
            "email"            : email,
            "age"              : age,
            "gender"           : random.choice(["Male", "Female", "Other", None]),
            "home_zone"        : home_zone,
            "work_zone"        : work_zone,
            "signup_date"      : signup.strftime("%Y-%m-%d"),
            "rating"           : rating,
            "total_rides"      : random.randint(0, 500),
            "preferred_vehicle": random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0],
            "is_prime_member"  : random.choice([True, False, None]),
        })

    df = pd.DataFrame(records)

    # ~1% duplicate rows — same user registered twice
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
        joined_date = datetime(2025, 1, 1) - timedelta(days=random.randint(30, 2500))

        vehicle_type = random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0]

        rating = round(np.random.normal(4.1, 0.4), 2)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.03:
            rating = None

        exp_years = round(
            (datetime(2025, 12, 31) - joined_date).days / 365, 1
        )

        home_zone = random.choice(ZONE_NAMES)

        vehicle_model = random.choice([
            "Maruti Swift", "Hyundai i20", "Honda City", "Toyota Etios",
            "Maruti Dzire", "Hyundai Xcent", "Bajaj RE Auto", "TVS Jupiter",
            "Honda Activa", "Royal Enfield", "Toyota Innova", None
        ])

        records.append({
            "driver_id"           : driver_id,
            "name"                : fake.name(),
            "phone"               : fake.phone_number() if random.random() > 0.02 else None,
            "home_zone"           : home_zone,
            "vehicle_type"        : vehicle_type,
            "vehicle_model"       : vehicle_model,
            "vehicle_year"        : random.choice([2016,2017,2018,2019,2020,2021,2022,2023,None]),
            "rating"              : rating,
            "total_rides"         : random.randint(50, 10000),
            "acceptance_rate"     : round(random.uniform(0.5, 1.0), 2) if random.random() > 0.05 else None,
            "cancellation_rate"   : round(random.uniform(0.0, 0.3), 2),
            "joined_date"         : joined_date.strftime("%Y-%m-%d"),
            "experience_years"    : exp_years,
            "is_active"           : random.choice([True, False]),
            "online_hours_per_day": round(random.uniform(2, 14), 1) if random.random() > 0.04 else None,
            "preferred_zones"     : random.choice(ZONE_NAMES),
        })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  TABLE 3 — WEATHER  (hourly per zone — Bangalore 2025)
# ================================================================

def generate_weather():
    """
    Hourly weather for each Bangalore zone for full year 2025.
    Weather varies by zone slightly — Whitefield and Electronic City
    are farther east/south and behave differently from central zones.
    Monsoon months (Jun-Sep) have significantly higher rain probability.
    """
    print("Generating weather table...")
    records = []

    conditions = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Storm", "Foggy"]

    start = datetime(2025, 1, 1, 0, 0)
    hours = 365 * 24

    # Generate for a representative subset of zones to keep file size reasonable
    # In production each zone would have its own sensor — we do 10 key zones
    key_zones = [
        "Koramangala", "Whitefield", "Electronic City", "Marathahalli",
        "MG Road", "Hebbal", "HSR Layout", "Indiranagar",
        "JP Nagar", "Yelahanka"
    ]

    for zone in key_zones:
        for h in range(hours):
            ts    = start + timedelta(hours=h)
            month = ts.month
            weights = get_weather_weights(month)
            condition  = random.choices(conditions, weights)[0]

            # Bangalore temperature ranges by season
            if month in {12, 1, 2}:
                temp = round(np.random.normal(20, 3), 1)    # cool winter
            elif month in {3, 4, 5}:
                temp = round(np.random.normal(33, 4), 1)    # hot summer
            elif month in MONSOON_MONTHS:
                temp = round(np.random.normal(25, 3), 1)    # cool monsoon
            else:
                temp = round(np.random.normal(27, 3), 1)    # post monsoon

            humidity = round(random.uniform(40, 95), 1)
            if month in MONSOON_MONTHS:
                humidity = round(random.uniform(70, 98), 1)

            wind_speed = round(random.uniform(0, 35), 1)

            # Sensor failures — 2% missing
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
    print("Generating rides table (main fact table)... this may take a moment")
    records = []

    driver_ids = drivers_df["driver_id"].tolist()
    user_ids   = users_df["user_id"].tolist()

    # Pre-index weather for fast lookup
    weather_index = {}
    for _, row in weather_df.iterrows():
        key = (row["zone"], row["timestamp"])
        weather_index[key] = row["condition"]

    # Zone names list for weather lookup (only key_zones have weather)
    weather_zones = weather_df["zone"].unique().tolist()

    for i in range(N_RIDES):
        ride_id      = f"RID{str(i+1).zfill(7)}"
        ts           = random_timestamp_2025()
        hour         = ts.hour
        month        = ts.month
        is_weekend   = ts.weekday() >= 5
        is_holiday   = ts.strftime("%Y-%m-%d") in INDIA_HOLIDAYS_2025

        pickup_zone  = random.choice(ZONE_NAMES)
        drop_zone    = random.choice(ZONE_NAMES)

        # Coordinates from zone bounds
        pickup_bounds = BANGALORE_ZONES[pickup_zone]
        drop_bounds   = BANGALORE_ZONES[drop_zone]
        pickup_lat    = round(random.uniform(*pickup_bounds["lat"]), 6)
        pickup_lon    = round(random.uniform(*pickup_bounds["lon"]), 6)
        drop_lat      = round(random.uniform(*drop_bounds["lat"]), 6)
        drop_lon      = round(random.uniform(*drop_bounds["lon"]), 6)

        vehicle_type  = random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0]

        # Distance — realistic Bangalore urban range
        distance_km   = round(np.random.exponential(scale=6), 2)
        distance_km   = float(np.clip(distance_km, 0.5, 45))
        if random.random() < 0.01:
            distance_km = random.choice([0, -1, 999])       # corrupt GPS

        # Weather lookup
        weather_hour = ts.strftime("%Y-%m-%d %H:00:00")
        lookup_zone  = pickup_zone if pickup_zone in weather_zones else random.choice(weather_zones)
        weather_cond = weather_index.get((lookup_zone, weather_hour), "Clear")

        # ── TARGET VARIABLE ──────────────────────────────────
        surge_multiplier = compute_surge_multiplier(
            hour, is_weekend, is_holiday,
            weather_cond, pickup_zone, month
        )

        # Fare calculation
        base_fare_map   = {"Auto": 30, "Mini": 50, "Sedan": 70, "SUV": 100, "Bike": 20}
        per_km_rate_map = {"Auto": 12, "Mini": 14, "Sedan": 17, "SUV":  22, "Bike": 8}
        fare = round(
            (base_fare_map[vehicle_type] + per_km_rate_map[vehicle_type] * distance_km)
            * surge_multiplier, 2
        )

        # Ride status
        status = random.choices(
            ["completed", "cancelled", "no_driver_found"],
            weights=[0.78, 0.17, 0.05]
        )[0]

        # Duration only for completed rides
        if status == "completed":
            duration_min = round(distance_km * random.uniform(3.0, 6.0), 1)
            if random.random() < 0.01:
                duration_min = random.choice([0, -5, 500])  # corrupt
        else:
            duration_min = None

        # Driver rating — only after completed rides, 25% skip rating
        driver_rating_given = None
        if status == "completed" and random.random() > 0.25:
            driver_rating_given = round(np.random.normal(4.0, 0.8), 1)
            driver_rating_given = float(np.clip(driver_rating_given, 1, 5))

        cancel_reason = None
        if status == "cancelled":
            cancel_reason = random.choice(CANCELLATION_REASONS)

        # App/backend failure nulls
        if random.random() < 0.03:
            pickup_zone = None
        if random.random() < 0.02:
            drop_zone = None
        if random.random() < 0.015:
            fare = None

        records.append({
            "ride_id"            : ride_id,
            "driver_id"          : random.choice(driver_ids),
            "user_id"            : random.choice(user_ids),
            "pickup_zone"        : pickup_zone,
            "drop_zone"          : drop_zone,
            "pickup_lat"         : pickup_lat,
            "pickup_lon"         : pickup_lon,
            "drop_lat"           : drop_lat,
            "drop_lon"           : drop_lon,
            "vehicle_type"       : vehicle_type,
            "ride_timestamp"     : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour"               : hour,
            "month"              : month,
            "day_of_week"        : ts.strftime("%A"),
            "is_weekend"         : is_weekend,
            "is_holiday"         : is_holiday,
            "weather_condition"  : weather_cond,
            "distance_km"        : distance_km,
            "duration_min"       : duration_min,
            "base_fare"          : base_fare_map[vehicle_type],
            "fare_amount"        : fare,
            "surge_multiplier"   : surge_multiplier,    # ← TARGET VARIABLE
            "ride_status"        : status,
            "cancellation_reason": cancel_reason,
            "driver_rating_given": driver_rating_given,
        })

    df = pd.DataFrame(records)

    # ~0.5% fully duplicate ride records — system glitch
    dupes = df.sample(frac=0.005, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)
    df    = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  TABLE 5 — PAYMENTS
# ================================================================

def generate_payments(rides_df):
    print("Generating payments table...")
    records = []

    completed = rides_df[rides_df["ride_status"] == "completed"].copy()

    for _, ride in completed.iterrows():
        payment_id = f"PAY{fake.unique.random_int(min=1000000, max=9999999)}"
        method     = random.choices(PAYMENT_METHODS, PAY_WEIGHTS)[0]

        pay_status = random.choices(
            ["success", "failed", "pending"],
            weights=[0.92, 0.05, 0.03]
        )[0]

        amount = ride["fare_amount"]
        tip    = None
        if amount is not None and pay_status == "success":
            tip = round(random.uniform(0, 50), 2) if random.random() < 0.2 else 0.0

        # Cash and failed payments have no transaction ID
        txn_id = fake.uuid4() if (method != "Cash" and pay_status != "failed") else None

        records.append({
            "payment_id"    : payment_id,
            "ride_id"       : ride["ride_id"],
            "payment_method": method,
            "amount"        : amount,
            "tip"           : tip,
            "payment_status": pay_status,
            "transaction_id": txn_id,
            "payment_time"  : ride["ride_timestamp"],
        })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  MAIN
# ================================================================

def main():
    print("=" * 60)
    print(" RealWorld-DS-ML | Ride Sharing | Surge Pricing")
    print(" City       : Bangalore")
    print(" Year       : 2025 (full year)")
    print(" Zones      : 25 real Bangalore zones")
    print(" Holidays   : India Public Holidays 2025")
    print("=" * 60)
    print(f" Rides      : {N_RIDES:,}")
    print(f" Drivers    : {N_DRIVERS:,}")
    print(f" Users      : {N_USERS:,}")
    print(f" Output dir : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    users_df    = generate_users()
    drivers_df  = generate_drivers()
    weather_df  = generate_weather()
    rides_df    = generate_rides(drivers_df, users_df, weather_df)
    payments_df = generate_payments(rides_df)

    users_df.to_csv(    os.path.join(OUTPUT_DIR, "users.csv"),    index=False)
    drivers_df.to_csv(  os.path.join(OUTPUT_DIR, "drivers.csv"),  index=False)
    weather_df.to_csv(  os.path.join(OUTPUT_DIR, "weather.csv"),  index=False)
    rides_df.to_csv(    os.path.join(OUTPUT_DIR, "rides.csv"),    index=False)
    payments_df.to_csv( os.path.join(OUTPUT_DIR, "payments.csv"), index=False)

    print()
    print("=" * 60)
    print(" ✅ All tables saved to data/raw/")
    print("=" * 60)
    print(f"  users.csv    : {len(users_df):>8,} rows")
    print(f"  drivers.csv  : {len(drivers_df):>8,} rows")
    print(f"  weather.csv  : {len(weather_df):>8,} rows")
    print(f"  rides.csv    : {len(rides_df):>8,} rows")
    print(f"  payments.csv : {len(payments_df):>8,} rows")
    print("=" * 60)
    print()
    print(" Next step → Open notebooks/01_data_generation.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
