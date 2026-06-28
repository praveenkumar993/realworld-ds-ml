"""
===============================================================
 RealWorld-DS-ML | Ride Sharing | ETA Prediction
 File    : data_generator.py
 Purpose : Generate realistic, production-style data simulating
           driver arrival time prediction for a Bangalore
           ride sharing platform (Ola/Uber/Rapido style).

 Tables Generated:
   1. assignments.csv — main fact table, one row per driver
                        assignment, target variable lives here
   2. drivers.csv     — driver profiles with speed behavior
   3. users.csv       — user profiles with pickup accuracy
   4. traffic.csv     — hourly traffic conditions per zone corridor

 Target Variable:
   actual_arrival_time_min — how many minutes the driver
   took to reach the user pickup location after assignment.
   This is what the Ola app shows: "Driver arrives in X min"

 Key Design Decisions:
   - Geographic realism: each zone has real lat/lon bounds,
     zone pairs have realistic road_to_straight ratios
   - Traffic realism: specific Bangalore corridors encoded
     with known peak hour behavior
   - Driver behavior: speed profiles, area familiarity
   - Weather realism: monsoon months significantly slower
   - Target generated from a physics-based formula:
     time = distance / speed + adjustments
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
N_ASSIGNMENTS = 60_000
N_DRIVERS     = 3_000
N_USERS       = 20_000

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Bangalore Zones with lat/lon bounds ──────────────────────
BANGALORE_ZONES = {
    "Koramangala"       : {"lat": (12.926, 12.940), "lon": (77.614, 77.632), "density": "high"},
    "Whitefield"        : {"lat": (12.960, 12.990), "lon": (77.730, 77.760), "density": "high"},
    "Indiranagar"       : {"lat": (12.970, 12.985), "lon": (77.635, 77.650), "density": "high"},
    "HSR Layout"        : {"lat": (12.905, 12.925), "lon": (77.630, 77.650), "density": "medium"},
    "Electronic City"   : {"lat": (12.830, 12.865), "lon": (77.660, 77.690), "density": "high"},
    "Marathahalli"      : {"lat": (12.945, 12.965), "lon": (77.695, 77.715), "density": "high"},
    "Bellandur"         : {"lat": (12.915, 12.935), "lon": (77.665, 77.685), "density": "medium"},
    "Sarjapur Road"     : {"lat": (12.880, 12.915), "lon": (77.670, 77.700), "density": "medium"},
    "BTM Layout"        : {"lat": (12.910, 12.928), "lon": (77.608, 77.625), "density": "high"},
    "Jayanagar"         : {"lat": (12.920, 12.940), "lon": (77.575, 77.595), "density": "medium"},
    "JP Nagar"          : {"lat": (12.890, 12.915), "lon": (77.570, 77.595), "density": "medium"},
    "Banashankari"      : {"lat": (12.895, 12.915), "lon": (77.545, 77.570), "density": "low"},
    "Rajajinagar"       : {"lat": (12.985, 13.005), "lon": (77.545, 77.565), "density": "medium"},
    "Malleswaram"       : {"lat": (13.000, 13.020), "lon": (77.560, 77.580), "density": "medium"},
    "Hebbal"            : {"lat": (13.030, 13.055), "lon": (77.585, 77.610), "density": "medium"},
    "Yelahanka"         : {"lat": (13.090, 13.120), "lon": (77.590, 77.620), "density": "low"},
    "Yeshwanthpur"      : {"lat": (13.015, 13.035), "lon": (77.535, 77.560), "density": "medium"},
    "Cunningham Road"   : {"lat": (12.990, 13.005), "lon": (77.590, 77.610), "density": "high"},
    "MG Road"           : {"lat": (12.970, 12.985), "lon": (77.600, 77.620), "density": "high"},
    "Shivajinagar"      : {"lat": (12.982, 12.998), "lon": (77.595, 77.615), "density": "high"},
    "KR Puram"          : {"lat": (13.000, 13.025), "lon": (77.685, 77.710), "density": "medium"},
    "Bannerghatta Road" : {"lat": (12.855, 12.890), "lon": (77.580, 77.610), "density": "low"},
    "Kadugodi"          : {"lat": (12.975, 12.998), "lon": (77.755, 77.780), "density": "low"},
    "Nagawara"          : {"lat": (13.040, 13.060), "lon": (77.615, 77.635), "density": "low"},
    "Vijayanagar"       : {"lat": (12.965, 12.985), "lon": (77.520, 77.545), "density": "low"},
}

ZONE_NAMES = list(BANGALORE_ZONES.keys())

# ── Zone Pair Traffic Profiles ────────────────────────────────
# Real Bangalore knowledge encoded:
# road_to_straight_ratio: how much longer the road is vs straight line
# base_traffic_multiplier: how congested this corridor typically is
# peak_multiplier: extra congestion during peak hours
ZONE_PAIR_PROFILES = {
    # Tech corridor nightmares
    ("Electronic City", "Koramangala")   : {"road_ratio": 1.45, "base_traffic": 1.8, "peak_mult": 2.5},
    ("Whitefield", "MG Road")            : {"road_ratio": 1.55, "base_traffic": 2.0, "peak_mult": 3.0},
    ("Marathahalli", "Indiranagar")      : {"road_ratio": 1.40, "base_traffic": 1.7, "peak_mult": 2.4},
    ("Bellandur", "HSR Layout")          : {"road_ratio": 1.30, "base_traffic": 1.5, "peak_mult": 2.0},
    ("KR Puram", "Whitefield")           : {"road_ratio": 1.35, "base_traffic": 1.6, "peak_mult": 2.2},
    ("Sarjapur Road", "BTM Layout")      : {"road_ratio": 1.40, "base_traffic": 1.7, "peak_mult": 2.3},
    # Central zone congestion
    ("Shivajinagar", "MG Road")          : {"road_ratio": 1.20, "base_traffic": 2.2, "peak_mult": 2.8},
    ("Cunningham Road", "Shivajinagar")  : {"road_ratio": 1.15, "base_traffic": 2.0, "peak_mult": 2.6},
    ("Malleswaram", "Rajajinagar")       : {"road_ratio": 1.25, "base_traffic": 1.6, "peak_mult": 2.0},
    # Outer ring — smoother
    ("Yelahanka", "Hebbal")              : {"road_ratio": 1.20, "base_traffic": 1.3, "peak_mult": 1.6},
    ("Banashankari", "JP Nagar")         : {"road_ratio": 1.15, "base_traffic": 1.2, "peak_mult": 1.5},
    ("Nagawara", "Hebbal")               : {"road_ratio": 1.18, "base_traffic": 1.3, "peak_mult": 1.7},
}

# Default profile for zone pairs not explicitly defined
DEFAULT_ZONE_PAIR = {"road_ratio": 1.30, "base_traffic": 1.5, "peak_mult": 2.0}

# ── Vehicle Type Speed Profiles (base speed in kmph) ─────────
VEHICLE_SPEED_PROFILES = {
    "Auto"  : {"free_flow": 22, "peak": 10, "rain_factor": 0.70},
    "Mini"  : {"free_flow": 28, "peak": 13, "rain_factor": 0.75},
    "Sedan" : {"free_flow": 32, "peak": 15, "rain_factor": 0.78},
    "SUV"   : {"free_flow": 35, "peak": 16, "rain_factor": 0.80},
    "Bike"  : {"free_flow": 30, "peak": 18, "rain_factor": 0.65},
}

VEHICLE_TYPES   = ["Auto",  "Mini",   "Sedan",  "SUV",    "Bike"]
VEHICLE_WEIGHTS = [0.20,     0.30,     0.25,     0.15,     0.10]

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

MONSOON_MONTHS = {6, 7, 8, 9}


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
    """
    Calculate straight-line distance between two GPS coordinates
    using the Haversine formula. Returns distance in kilometers.

    The Haversine formula accounts for the curvature of the Earth.
    For short distances within a city, this is very accurate.

    Parameters:
        lat1, lon1: coordinates of point 1 (driver location)
        lat2, lon2: coordinates of point 2 (pickup location)
    Returns:
        distance in kilometers (straight line)
    """
    R = 6371  # Earth's radius in km

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def get_zone_pair_profile(driver_zone, pickup_zone):
    """
    Get traffic profile for a zone pair.
    Checks both directions (A→B and B→A) since traffic
    is the same regardless of direction in our simulation.
    Falls back to default if pair not explicitly defined.
    """
    pair1 = (driver_zone, pickup_zone)
    pair2 = (pickup_zone, driver_zone)

    if pair1 in ZONE_PAIR_PROFILES:
        return ZONE_PAIR_PROFILES[pair1]
    elif pair2 in ZONE_PAIR_PROFILES:
        return ZONE_PAIR_PROFILES[pair2]
    else:
        return DEFAULT_ZONE_PAIR.copy()


def get_weather_condition(month, hour):
    """
    Get a realistic weather condition for Bangalore
    based on month (monsoon vs non-monsoon) and hour.
    """
    conditions = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Storm", "Foggy"]

    if month in MONSOON_MONTHS:
        # Monsoon: more rain, especially afternoon/evening
        if hour in range(14, 22):
            weights = [0.15, 0.20, 0.30, 0.22, 0.08, 0.05]
        else:
            weights = [0.25, 0.30, 0.22, 0.12, 0.05, 0.06]
    elif month in {11, 12, 1, 2}:
        # Winter: mostly clear, some fog in early morning
        if hour in range(4, 8):
            weights = [0.40, 0.25, 0.05, 0.02, 0.01, 0.27]
        else:
            weights = [0.60, 0.28, 0.06, 0.02, 0.01, 0.03]
    else:
        # Normal months
        weights = [0.50, 0.28, 0.12, 0.06, 0.02, 0.02]

    return random.choices(conditions, weights)[0]


def compute_traffic_multiplier(hour, is_weekend, is_holiday,
                                zone_pair_profile, weather):
    """
    Compute the traffic multiplier for a specific situation.
    Higher multiplier = more congestion = slower speed = longer ETA.

    Base is the zone pair's inherent congestion level.
    Then adjusted for:
    - Peak vs off-peak hours
    - Weekend (less office traffic)
    - Holiday (very low traffic)
    - Weather (rain causes massive slowdowns in Bangalore)
    """
    base  = zone_pair_profile["base_traffic"]
    peak  = zone_pair_profile["peak_mult"]

    # Time-based traffic
    if hour in range(8, 11):
        # Morning peak
        traffic = base * random.uniform(1.20, 1.45)
    elif hour in range(17, 21):
        # Evening peak
        traffic = base * random.uniform(1.25, 1.50)
    elif hour in range(11, 17):
        # Midday — moderate
        traffic = base * random.uniform(0.85, 1.05)
    elif hour in range(21, 24) or hour in range(0, 6):
        # Night — light traffic
        traffic = base * random.uniform(0.35, 0.50)
    else:
        # Early morning
        traffic = base * random.uniform(0.45, 0.60)

    # Weekend reduces peak traffic (offices closed)
    if is_weekend and hour in range(8, 21):
        traffic *= random.uniform(0.65, 0.80)

    # Holiday further reduces traffic
    if is_holiday:
        traffic *= random.uniform(0.50, 0.65)

    # Weather — Bangalore rain adds congestion but not extreme
    if weather == "Storm":
        traffic *= random.uniform(1.20, 1.45)
    elif weather == "Heavy Rain":
        traffic *= random.uniform(1.10, 1.30)
    elif weather == "Light Rain":
        traffic *= random.uniform(1.05, 1.15)
    elif weather == "Foggy":
        traffic *= random.uniform(1.05, 1.10)

    # Clip to realistic range
    return round(np.clip(traffic, 0.5, 3.5), 2)


def compute_actual_arrival_time(
    straight_line_distance_km,
    road_distance_km,
    vehicle_type,
    traffic_multiplier,
    weather,
    driver_familiarity_score,
    pickup_pin_accuracy,
    hour,
):
    """
    Core formula: compute how long the driver actually takes
    to reach the pickup location.

    Formula:
        base_time = road_distance / effective_speed × 60

    Effective speed is reduced by:
        - Traffic multiplier (congestion)
        - Weather (rain slows everything)
        - Night visibility

    Then adjustments added for:
        + Driver familiarity (knows shortcuts → faster)
        + Pin accuracy penalty (bad pin → driver circles)
        + Random noise (real life variance)

    Returns actual arrival time in minutes.
    """
    speed_profile = VEHICLE_SPEED_PROFILES[vehicle_type]

    # Base speed — free flow or peak depending on traffic
    if traffic_multiplier > 2.0:
        base_speed = speed_profile["peak"]
    else:
        base_speed = speed_profile["free_flow"]

    # Reduce speed for weather
    if weather in ["Heavy Rain", "Storm"]:
        base_speed *= speed_profile["rain_factor"]
    elif weather == "Light Rain":
        base_speed *= (speed_profile["rain_factor"] + 1) / 2
    elif weather == "Foggy":
        base_speed *= 0.85

    # Reduce speed for traffic
    effective_speed = base_speed / traffic_multiplier

    # Minimum realistic speed — even gridlock moves at ~5 kmph
    effective_speed = max(effective_speed, 4.0)

    # Base travel time in minutes
    base_time_min = (road_distance_km / effective_speed) * 60

    # Driver familiarity bonus (0.0-1.0 score)
    # A driver who knows the area well saves up to 3 minutes
    familiarity_saving = driver_familiarity_score * random.uniform(0, 3.0)
    base_time_min -= familiarity_saving

    # Pickup pin accuracy penalty
    # A poorly placed pin means driver circles for extra time
    # accuracy=1.0 → no penalty, accuracy=0.3 → up to 4 min penalty
    pin_penalty = (1.0 - pickup_pin_accuracy) * random.uniform(0, 4.0)
    base_time_min += pin_penalty

    # Late night adjustment — empty roads but driver may be cautious
    if hour in range(0, 5):
        base_time_min *= random.uniform(0.75, 0.90)

    # Random noise — real life is never perfectly predictable
    # Accounts for traffic signal timing, pedestrians, etc.
    noise = np.random.normal(0, 1.2)
    base_time_min += noise

    # Clip to realistic range: minimum 1 minute, maximum 60 minutes
    actual_time = round(np.clip(base_time_min, 1.0, 60.0), 1)
    return actual_time


# ================================================================
#  TABLE 1 — DRIVERS
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

        # Speed profile — some drivers are naturally faster
        speed_style = random.choices(
            ["aggressive", "normal", "cautious"],
            weights=[0.20, 0.60, 0.20]
        )[0]

        if speed_style == "aggressive":
            speed_factor = round(random.uniform(1.10, 1.25), 2)
        elif speed_style == "normal":
            speed_factor = round(random.uniform(0.90, 1.10), 2)
        else:
            speed_factor = round(random.uniform(0.75, 0.90), 2)

        # Area familiarity — 0.0 to 1.0
        # Drivers who have been on the platform longer know more areas
        exp_years = (datetime(2025, 12, 31) - joined_date).days / 365
        familiarity_base = min(0.9, exp_years / 5.0)
        area_familiarity = round(
            np.clip(familiarity_base + random.uniform(-0.2, 0.2), 0.1, 1.0), 2
        )

        # Home zone — drivers are fastest in their home zone
        home_zone = random.choice(ZONE_NAMES)

        rating = round(np.random.normal(4.1, 0.4), 2)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.03:
            rating = None

        records.append({
            "driver_id"            : driver_id,
            "name"                 : fake.name(),
            "phone"                : fake.phone_number() if random.random() > 0.02 else None,
            "vehicle_type"         : vehicle_type,
            "vehicle_model"        : random.choice([
                "Maruti Swift","Hyundai i20","Honda City","Toyota Etios",
                "Maruti Dzire","Bajaj RE Auto","TVS Jupiter",
                "Honda Activa","Toyota Innova", None
            ]),
            "vehicle_year"         : random.choice([2016,2017,2018,2019,2020,2021,2022,2023,None]),
            "rating"               : rating,
            "speed_style"          : speed_style,
            "speed_factor"         : speed_factor,
            "area_familiarity"     : area_familiarity,
            "home_zone"            : home_zone,
            "joined_date"          : joined_date.strftime("%Y-%m-%d"),
            "experience_years"     : round(exp_years, 1),
            "total_rides"          : random.randint(50, 12000),
            "is_active"            : random.choice([True, False]),
            "online_hours_per_day" : round(random.uniform(2, 14), 1) if random.random() > 0.04 else None,
            "avg_speed_kmph"       : round(
                VEHICLE_SPEED_PROFILES[vehicle_type]["free_flow"] * speed_factor, 1
            ),
        })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  TABLE 2 — USERS
# ================================================================

def generate_users():
    print("Generating users table...")
    records = []

    for i in range(N_USERS):
        user_id = f"USR{str(i+1).zfill(6)}"
        signup  = datetime(2025, 1, 1) - timedelta(
            days=random.randint(0, 1460)
        )

        # Pickup pin accuracy — how precisely user drops the pin
        # Regular users are better at this over time
        total_rides = random.randint(0, 500)
        if total_rides > 100:
            pin_accuracy = round(random.uniform(0.65, 1.0), 2)
        elif total_rides > 20:
            pin_accuracy = round(random.uniform(0.50, 0.85), 2)
        else:
            pin_accuracy = round(random.uniform(0.30, 0.70), 2)

        if random.random() < 0.04:
            pin_accuracy = None

        age = random.randint(18, 60)
        if random.random() < 0.02:
            age = random.choice([-1, 0, 150, 999])

        rating = round(np.random.normal(4.2, 0.5), 1)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.04:
            rating = None

        records.append({
            "user_id"          : user_id,
            "name"             : fake.name(),
            "phone"            : fake.phone_number() if random.random() > 0.03 else None,
            "email"            : fake.email() if random.random() > 0.05 else None,
            "age"              : age,
            "gender"           : random.choice(["Male","Female","Other",None]),
            "home_zone"        : random.choice(ZONE_NAMES),
            "work_zone"        : random.choice(ZONE_NAMES),
            "signup_date"      : signup.strftime("%Y-%m-%d"),
            "rating"           : rating,
            "total_rides"      : total_rides,
            "pickup_pin_accuracy": pin_accuracy,
            "preferred_vehicle": random.choices(VEHICLE_TYPES, VEHICLE_WEIGHTS)[0],
            "is_prime_member"  : random.choice([True, False, None]),
        })

    df = pd.DataFrame(records)
    dupes = df.sample(frac=0.01, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)

    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  TABLE 3 — TRAFFIC  (hourly per zone corridor)
# ================================================================

def generate_traffic():
    """
    Hourly traffic conditions per major zone corridor.
    This simulates the external traffic signal feed that
    a real ETA system would pull from HERE Maps or Google Maps.
    """
    print("Generating traffic table...")
    records = []

    # Generate for key zone corridors — the high traffic ones
    key_corridors = list(ZONE_PAIR_PROFILES.keys())

    start = datetime(2025, 1, 1, 0, 0)
    hours = 365 * 24

    for (zone_a, zone_b) in key_corridors:
        profile = ZONE_PAIR_PROFILES[(zone_a, zone_b)]

        for h in range(hours):
            ts    = start + timedelta(hours=h)
            hour  = ts.hour
            month = ts.month
            is_weekend = ts.weekday() >= 5
            is_holiday = ts.strftime("%Y-%m-%d") in INDIA_HOLIDAYS_2025
            weather    = get_weather_condition(month, hour)

            traffic_mult = compute_traffic_multiplier(
                hour, is_weekend, is_holiday,
                profile, weather
            )

            # Average speed on this corridor at this time
            avg_speed = round(
                random.uniform(20, 35) / traffic_mult, 1
            )
            avg_speed = max(avg_speed, 4.0)

            records.append({
                "zone_a"            : zone_a,
                "zone_b"            : zone_b,
                "timestamp"         : ts.strftime("%Y-%m-%d %H:00:00"),
                "hour"              : hour,
                "month"             : month,
                "is_weekend"        : is_weekend,
                "weather_condition" : weather,
                "traffic_multiplier": traffic_mult,
                "avg_speed_kmph"    : avg_speed,
            })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | {df.isnull().sum().sum():,} null values injected")
    return df


# ================================================================
#  TABLE 4 — ASSIGNMENTS  (main fact table)
# ================================================================

def generate_assignments(drivers_df, users_df):
    print("Generating assignments table (main fact table)... this may take a moment")
    records = []

    # Build fast lookup dicts
    driver_lookup = (
        drivers_df.drop_duplicates(subset=["driver_id"])
                  .set_index("driver_id")
                  .to_dict("index")
    )
    user_lookup = (
        users_df.drop_duplicates(subset=["user_id"])
                .set_index("user_id")
                .to_dict("index")
    )

    driver_ids = drivers_df["driver_id"].unique().tolist()
    user_ids   = users_df["user_id"].unique().tolist()

    for i in range(N_ASSIGNMENTS):
        assignment_id = f"ASN{str(i+1).zfill(7)}"

        # ── Time ──────────────────────────────────────────────
        ts         = random_timestamp_2025()
        hour       = ts.hour
        month      = ts.month
        is_weekend = ts.weekday() >= 5
        is_holiday = ts.strftime("%Y-%m-%d") in INDIA_HOLIDAYS_2025
        weather    = get_weather_condition(month, hour)

        # ── Zone assignment ────────────────────────────────────
        # Driver zone and pickup zone
        # 60% of time driver is in the same zone as pickup
        # 40% of time driver is in a different zone nearby
        pickup_zone = random.choice(ZONE_NAMES)
        if random.random() < 0.60:
            driver_zone = pickup_zone
        else:
            driver_zone = random.choice(ZONE_NAMES)

        driver_zone_info = BANGALORE_ZONES[driver_zone]
        pickup_zone_info = BANGALORE_ZONES[pickup_zone]

        # ── GPS Coordinates ────────────────────────────────────
        driver_lat = round(random.uniform(*driver_zone_info["lat"]), 6)
        driver_lon = round(random.uniform(*driver_zone_info["lon"]), 6)
        pickup_lat = round(random.uniform(*pickup_zone_info["lat"]), 6)
        pickup_lon = round(random.uniform(*pickup_zone_info["lon"]), 6)

        # ── Haversine distance ─────────────────────────────────
        straight_line_km = round(
            haversine_distance(driver_lat, driver_lon, pickup_lat, pickup_lon), 3
        )

        # ── Road distance ──────────────────────────────────────
        # Road distance is always longer than straight line
        zone_pair_profile = get_zone_pair_profile(driver_zone, pickup_zone)
        road_ratio = zone_pair_profile["road_ratio"]

        # Add some randomness to road ratio
        road_ratio_actual = road_ratio * random.uniform(0.90, 1.10)
        road_distance_km  = round(straight_line_km * road_ratio_actual, 3)

        # ── Vehicle and Driver ──────────────────────────────────
        driver_id  = random.choice(driver_ids)
        driver_row = driver_lookup.get(driver_id, {})
        vehicle_type       = driver_row.get("vehicle_type", "Mini")
        speed_factor       = driver_row.get("speed_factor", 1.0)
        area_familiarity   = driver_row.get("area_familiarity", 0.5)
        driver_home_zone   = driver_row.get("home_zone", pickup_zone)
        driver_rating      = driver_row.get("rating", 4.0)

        # Driver knows pickup zone better if it's their home zone
        familiarity_score = area_familiarity
        if driver_home_zone == pickup_zone:
            familiarity_score = min(1.0, area_familiarity + 0.2)
        if random.random() < 0.03:
            familiarity_score = None

        # ── User ──────────────────────────────────────────────
        user_id  = random.choice(user_ids)
        user_row = user_lookup.get(user_id, {})
        pickup_pin_accuracy = user_row.get("pickup_pin_accuracy", 0.7)
        user_rating         = user_row.get("rating", 4.2)

        if pickup_pin_accuracy is None:
            pickup_pin_accuracy = 0.70

        # ── Traffic multiplier ─────────────────────────────────
        traffic_multiplier = compute_traffic_multiplier(
            hour, is_weekend, is_holiday,
            zone_pair_profile, weather
        )

        # ── Zone density (affects search time in dense areas) ──
        pickup_density = pickup_zone_info["density"]
        density_score  = {"high": 0.8, "medium": 0.5, "low": 0.2}[pickup_density]

        # ── Surge at assignment time ───────────────────────────
        # Surge affects driver motivation but not ETA directly
        surge = 1.0
        if hour in range(8, 11) or hour in range(17, 21):
            surge += random.uniform(0.3, 0.8)
        if weather in ["Heavy Rain", "Storm"]:
            surge += random.uniform(0.2, 0.6)
        surge = round(np.clip(surge, 1.0, 4.5), 1)

        # ── TARGET VARIABLE ───────────────────────────────────
        actual_arrival_time = compute_actual_arrival_time(
            straight_line_distance_km = straight_line_km,
            road_distance_km          = road_distance_km,
            vehicle_type              = vehicle_type,
            traffic_multiplier        = traffic_multiplier,
            weather                   = weather,
            driver_familiarity_score  = familiarity_score if familiarity_score else 0.5,
            pickup_pin_accuracy       = pickup_pin_accuracy,
            hour                      = hour,
        )

        # Apply driver speed factor (faster/slower than average)
        actual_arrival_time = round(
            actual_arrival_time / (speed_factor if speed_factor else 1.0), 1
        )
        actual_arrival_time = float(np.clip(actual_arrival_time, 1.0, 60.0))

        # ── App-shown ETA ──────────────────────────────────────
        # The app's estimate — systematically slightly optimistic
        # Companies deliberately show slightly lower ETA
        app_bias = random.uniform(0.80, 0.95)
        app_shown_eta = round(actual_arrival_time * app_bias, 1)
        if random.random() < 0.02:
            app_shown_eta = None

        # ── Intentional messiness ──────────────────────────────
        if random.random() < 0.02:
            straight_line_km = None
        if random.random() < 0.015:
            road_distance_km = None
        if random.random() < 0.025:
            traffic_multiplier = None
        if random.random() < 0.03:
            pickup_pin_accuracy = None

        records.append({
            # IDs
            "assignment_id"              : assignment_id,
            "driver_id"                  : driver_id,
            "user_id"                    : user_id,

            # Time
            "assignment_timestamp"       : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour"                       : hour,
            "month"                      : month,
            "day_of_week"                : ts.strftime("%A"),
            "is_weekend"                 : is_weekend,
            "is_holiday"                 : is_holiday,

            # Geography
            "driver_zone"                : driver_zone,
            "pickup_zone"                : pickup_zone,
            "driver_lat"                 : driver_lat,
            "driver_lon"                 : driver_lon,
            "pickup_lat"                 : pickup_lat,
            "pickup_lon"                 : pickup_lon,
            "straight_line_distance_km"  : straight_line_km,
            "road_distance_km"           : road_distance_km,
            "road_to_straight_ratio"     : round(road_ratio_actual, 3),

            # Traffic and conditions
            "weather_condition"          : weather,
            "traffic_multiplier"         : traffic_multiplier,
            "pickup_zone_density"        : pickup_density,
            "density_score"              : density_score,
            "surge_at_assignment"        : surge,

            # Driver signals
            "vehicle_type"               : vehicle_type,
            "driver_speed_factor"        : speed_factor,
            "driver_familiarity_score"   : familiarity_score,
            "driver_rating"              : driver_rating,
            "driver_home_zone"           : driver_home_zone,
            "is_driver_in_home_zone"     : int(driver_home_zone == pickup_zone),

            # User signals
            "pickup_pin_accuracy"        : pickup_pin_accuracy,
            "user_rating"                : user_rating,

            # App prediction
            "app_shown_eta_min"          : app_shown_eta,

            # TARGET VARIABLE
            "actual_arrival_time_min"    : actual_arrival_time,
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
    print(" RealWorld-DS-ML | Ride Sharing | ETA Prediction")
    print(" City         : Bangalore")
    print(" Year         : 2025 (full year)")
    print(" Zones        : 25 real Bangalore zones")
    print(" Target       : actual_arrival_time_min")
    print(" Problem Type : Regression")
    print("=" * 62)
    print(f" Assignments  : {N_ASSIGNMENTS:,}")
    print(f" Drivers      : {N_DRIVERS:,}")
    print(f" Users        : {N_USERS:,}")
    print(f" Output dir   : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 62)

    drivers_df     = generate_drivers()
    users_df       = generate_users()
    traffic_df     = generate_traffic()
    assignments_df = generate_assignments(drivers_df, users_df)

    drivers_df.to_csv(    os.path.join(OUTPUT_DIR, "drivers.csv"),     index=False)
    users_df.to_csv(      os.path.join(OUTPUT_DIR, "users.csv"),       index=False)
    traffic_df.to_csv(    os.path.join(OUTPUT_DIR, "traffic.csv"),     index=False)
    assignments_df.to_csv(os.path.join(OUTPUT_DIR, "assignments.csv"), index=False)

    print()
    print("=" * 62)
    print(" ✅ All tables saved to data/raw/")
    print("=" * 62)
    print(f"  drivers.csv     : {len(drivers_df):>8,} rows")
    print(f"  users.csv       : {len(users_df):>8,} rows")
    print(f"  traffic.csv     : {len(traffic_df):>8,} rows")
    print(f"  assignments.csv : {len(assignments_df):>8,} rows")
    print("=" * 62)

    # ── Target variable summary ───────────────────────────────
    eta = assignments_df["actual_arrival_time_min"]
    print()
    print(" Target Variable — actual_arrival_time_min:")
    print(f"   Mean   : {eta.mean():.2f} min")
    print(f"   Median : {eta.median():.2f} min")
    print(f"   Std    : {eta.std():.2f} min")
    print(f"   Min    : {eta.min():.2f} min")
    print(f"   Max    : {eta.max():.2f} min")
    print()
    print("   Distribution by bucket:")
    buckets = pd.cut(eta, bins=[0, 5, 10, 15, 20, 30, 60],
                     labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30+"])
    for label, count in buckets.value_counts().sort_index().items():
        pct = count / len(eta) * 100
        print(f"     {label:<8} min : {count:>7,}  ({pct:.1f}%)")
    print("=" * 62)
    print()
    print(" Next step → Open notebooks/01_data_generation.ipynb")
    print("=" * 62)


if __name__ == "__main__":
    main()