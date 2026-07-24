"""
===============================================================
 RealWorld-DS-ML | Ride Sharing | Driver Churn Prediction
 File    : data_generator.py
 Purpose : Generate realistic, production-style data simulating
           driver churn behavior on a Bangalore ride sharing
           platform (Ola/Uber/Rapido style).

 Tables Generated:
   1. drivers.csv         — driver profiles (one row per driver)
   2. weekly_activity.csv — weekly behavioral signals per driver
                            (one row per driver per week)
   3. incentives.csv      — incentive/bonus offers sent to drivers
   4. support_tickets.csv — driver complaints and support history

 Target Variable:
   is_churned: 1 = driver went inactive in observation window
               0 = driver remained active

 Churn Definition:
   Active window    : Weeks 1-12  (Jan - Mar 2025)
   Observation window: Weeks 13-16 (Apr 2025)
   Churned          : Active in weeks 1-12 but ZERO rides in 13-16
   Retained         : Active in both windows

 Key Design Decisions:
   - Time-series behavioral data first, then aggregated
   - Churn is not a transaction event — it is ABSENCE of events
   - Trend features: is ride volume going up or down over time?
   - RFM framework: Recency, Frequency, Monetary value
   - Rolling window: last 4 weeks vs weeks 1-8
   - Multiple churn drivers: earnings drop, cancellation rate rise,
     competition, personal reasons, vehicle issues
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

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

# ── Config ────────────────────────────────────────────────────
N_DRIVERS = 5_000   # more drivers for churn — need enough churners

OUTPUT_DIR = os.path.join(os.path.dirname(__file__),
                          "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Bangalore Zones ───────────────────────────────────────────
BANGALORE_ZONES = [
    "Koramangala", "Whitefield", "Indiranagar", "HSR Layout",
    "Electronic City", "Marathahalli", "Bellandur", "Sarjapur Road",
    "BTM Layout", "Jayanagar", "JP Nagar", "Banashankari",
    "Rajajinagar", "Malleswaram", "Hebbal", "Yelahanka",
    "Yeshwanthpur", "Cunningham Road", "MG Road", "Shivajinagar",
    "KR Puram", "Bannerghatta Road", "Kadugodi", "Nagawara",
    "Vijayanagar",
]

VEHICLE_TYPES   = ["Auto", "Mini", "Sedan", "SUV", "Bike"]
VEHICLE_WEIGHTS = [0.20,   0.30,   0.25,   0.15,  0.10]

# ── Vehicle base earnings per ride ───────────────────────────
VEHICLE_EARNINGS = {
    "Auto": 80,  "Mini": 120, "Sedan": 160,
    "SUV": 220,  "Bike": 50
}

# ── Churn risk profiles ───────────────────────────────────────
# Each driver starts with a base churn risk that evolves
# based on their experience and weekly signals
CHURN_PROFILES = {
    "stable"      : {"base_risk": 0.05, "weight": 0.40},
    "at_risk"     : {"base_risk": 0.35, "weight": 0.30},
    "high_risk"   : {"base_risk": 0.65, "weight": 0.20},
    "new_driver"  : {"base_risk": 0.45, "weight": 0.10},
}

# ── Churn reasons ─────────────────────────────────────────────
CHURN_REASONS = [
    "earnings_drop",      # income fell below expectations
    "competitor_platform",# switched to competitor
    "vehicle_issue",      # car/bike broke down
    "personal_reasons",   # health, family, other work
    "high_cancellations", # too many rider cancellations
    "incentive_removed",  # platform removed bonus program
    "burnout",            # too many hours, fatigue
    "unknown",            # driver simply stopped without reason
]

# ── India Public Holidays 2025 (Jan-Apr) ─────────────────────
INDIA_HOLIDAYS_Q1_2025 = {
    "2025-01-01", "2025-01-14", "2025-01-26",
    "2025-02-26", "2025-03-14", "2025-03-31",
}


# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def week_to_date(week_num):
    """Convert week number (1-16) to approximate date in 2025."""
    start = datetime(2025, 1, 1)
    return start + timedelta(weeks=week_num - 1)


def compute_weekly_earnings(rides, vehicle_type,
                            surge_factor, cancellation_rate):
    """
    Compute weekly earnings for a driver.
    Earnings = rides × base_earning × surge × (1 - cancel_rate)
    Cancelled rides earn nothing.
    """
    base = VEHICLE_EARNINGS.get(vehicle_type, 120)
    completed_rides = max(0, int(rides * (1 - cancellation_rate)))
    earnings = completed_rides * base * surge_factor
    # Add tip (20% of drivers consistently get tips)
    tip_factor = random.uniform(1.0, 1.15)
    return round(earnings * tip_factor, 2)


# ================================================================
#  TABLE 1 — DRIVERS
# ================================================================

def generate_drivers():
    """
    Driver profile table. One row per driver.
    Contains demographic, vehicle, and historical behavioral info.
    """
    print("Generating drivers table...")
    records = []

    for i in range(N_DRIVERS):
        driver_id = f"DRV{str(i+1).zfill(5)}"

        # Join date — drivers who joined recently have different
        # churn patterns than experienced drivers
        days_on_platform = random.randint(30, 1460)
        join_date = datetime(2025, 1, 1) - timedelta(
            days=days_on_platform
        )

        vehicle_type = random.choices(
            VEHICLE_TYPES, VEHICLE_WEIGHTS
        )[0]

        # Churn profile — determines behavioral trajectory
        profile_name = random.choices(
            list(CHURN_PROFILES.keys()),
            weights=[v["weight"]
                     for v in CHURN_PROFILES.values()]
        )[0]

        # Driver demographics
        age = random.randint(21, 55)
        gender = random.choices(
            ["Male", "Female", "Other"],
            weights=[0.88, 0.10, 0.02]
        )[0]

        # Rating — at-risk drivers often have slightly lower ratings
        if profile_name == "stable":
            rating = round(np.random.normal(4.3, 0.3), 2)
        elif profile_name == "at_risk":
            rating = round(np.random.normal(4.0, 0.4), 2)
        else:
            rating = round(np.random.normal(3.8, 0.5), 2)
        rating = float(np.clip(rating, 1.0, 5.0))
        if random.random() < 0.03:
            rating = None

        # Whether driver has another income source
        # Drivers with other income churn more easily
        has_other_income = random.random() < 0.35

        # City knowledge — drivers who know more zones
        # have higher earnings and lower churn
        zones_known = random.randint(3, 25)

        # Platform — some drivers also use competing apps
        uses_competitor = random.random() < 0.40

        # Bank account linked — unlinked = payment issues = churn risk
        bank_linked = random.random() > 0.05
        if random.random() < 0.02:
            bank_linked = None

        records.append({
            "driver_id"          : driver_id,
            "name"               : fake.name(),
            "age"                : age,
            "gender"             : gender,
            "phone"              : fake.phone_number()
                                   if random.random() > 0.02 else None,
            "vehicle_type"       : vehicle_type,
            "vehicle_year"       : random.choice(
                [2016,2017,2018,2019,2020,2021,2022,2023,None]
            ),
            "home_zone"          : random.choice(BANGALORE_ZONES),
            "join_date"          : join_date.strftime("%Y-%m-%d"),
            "days_on_platform"   : days_on_platform,
            "experience_years"   : round(days_on_platform / 365, 1),
            "churn_profile"      : profile_name,
            "rating"             : rating,
            "has_other_income"   : has_other_income,
            "zones_known"        : zones_known,
            "uses_competitor_app": uses_competitor,
            "bank_account_linked": bank_linked,
            "is_prime_driver"    : random.random() < 0.25,
            "referral_source"    : random.choice([
                "friend_referral", "social_media",
                "offline_ad", "existing_driver", None
            ]),
        })

    df = pd.DataFrame(records)
    # Inject duplicates
    dupes = df.sample(frac=0.01, random_state=SEED).copy()
    df    = pd.concat([df, dupes], ignore_index=True)

    print(f"  → {len(df):,} rows | "
          f"{df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  TABLE 2 — WEEKLY ACTIVITY
# ================================================================

def generate_weekly_activity(drivers_df):
    """
    Weekly behavioral time series for each driver.
    Weeks 1-12 = active window (training period)
    Weeks 13-16 = observation window (churn determination)

    This is the most important table in the project —
    it captures how driver behavior CHANGES over time,
    which is what churn prediction models learn from.

    Churn pattern: At-risk drivers show declining ride counts,
    rising cancellation rates, and falling earnings in the
    weeks before they eventually go silent.
    """
    print("Generating weekly activity table... "
          "(this takes a moment)")
    records = []

    # Build driver lookup
    driver_lookup = (
        drivers_df.drop_duplicates(subset=["driver_id"])
                  .set_index("driver_id")
                  .to_dict("index")
    )
    driver_ids = list(driver_lookup.keys())

    for driver_id in driver_ids:
        dr = driver_lookup[driver_id]
        profile  = dr.get("churn_profile", "stable")
        vehicle  = dr.get("vehicle_type", "Mini")
        rating   = dr.get("rating", 4.0) or 4.0
        exp_yrs  = dr.get("experience_years", 1.0)

        # Base weekly rides for this driver
        # Experienced drivers do more rides
        if exp_yrs > 2:
            base_rides = random.randint(25, 60)
        elif exp_yrs > 1:
            base_rides = random.randint(15, 40)
        else:
            base_rides = random.randint(8, 25)

        # Will this driver churn?
        # Churn probability driven by profile + other factors
        base_churn_prob = CHURN_PROFILES[profile]["base_risk"]

        # Adjust for experience — very new and very experienced churn less
        if exp_yrs < 0.25:
            base_churn_prob += 0.15  # very new drivers often quit
        elif exp_yrs > 3:
            base_churn_prob -= 0.10  # veterans are loyal

        # Other income means easier to leave
        if dr.get("has_other_income", False):
            base_churn_prob += 0.10

        # Competitor app user = higher risk
        if dr.get("uses_competitor_app", False):
            base_churn_prob += 0.08

        # Low rating = more cancellations = frustration = churn
        if rating < 3.8:
            base_churn_prob += 0.12

        # Unlinked bank = payment issues = churn
        if not dr.get("bank_account_linked", True):
            base_churn_prob += 0.20

        base_churn_prob = float(np.clip(base_churn_prob, 0.02, 0.95))
        will_churn      = random.random() < base_churn_prob

        # Churn reason
        if will_churn:
            churn_reason = random.choices(
                CHURN_REASONS,
                weights=[0.25, 0.15, 0.12, 0.12,
                         0.10, 0.10, 0.08, 0.08]
            )[0]
        else:
            churn_reason = None

        # Generate 16 weeks of activity
        # Churning drivers show decline starting around week 9-11
        churn_start_week = random.randint(8, 12) if will_churn else 99

        for week in range(1, 17):
            week_date = week_to_date(week)

            # ── Rides this week ───────────────────────────────
            # Stable drivers have consistent ride counts
            # At-risk drivers show decline before churning
            weeks_to_churn = churn_start_week - week

            if will_churn and week > churn_start_week:
                # Post-churn — zero rides
                rides_this_week = 0
            elif will_churn and weeks_to_churn <= 3:
                # Pre-churn decline — rides dropping
                decline_factor = max(0.1, weeks_to_churn / 3.0)
                rides_this_week = int(
                    base_rides * decline_factor
                    * random.uniform(0.5, 0.9)
                )
            else:
                # Normal active period
                # Add seasonality — Jan-Feb slightly slower
                seasonal = 0.90 if week_date.month in [1, 2] else 1.0

                # Weekend boost — some drivers work more on weekends
                week_noise = random.uniform(0.75, 1.25)
                rides_this_week = int(
                    base_rides * seasonal * week_noise
                )

            rides_this_week = max(0, rides_this_week)

            # ── Hours online ──────────────────────────────────
            # Assume roughly 20-25 min per ride on average
            if rides_this_week > 0:
                hours_online = round(
                    rides_this_week * random.uniform(0.35, 0.55)
                    + random.uniform(0, 3), 1
                )
            else:
                hours_online = round(random.uniform(0, 2), 1) \
                    if (will_churn and weeks_to_churn <= 1) \
                    else 0.0

            # ── Cancellation rate ─────────────────────────────
            # Churning drivers often show rising cancel rates
            # (frustration, less care about metrics)
            base_cancel = 0.08 if profile == "stable" else 0.15
            if will_churn and weeks_to_churn <= 4:
                cancel_rate = round(
                    min(0.50, base_cancel
                        + (4 - weeks_to_churn) * 0.05
                        + random.uniform(0, 0.05)), 3
                )
            else:
                cancel_rate = round(
                    base_cancel + random.uniform(-0.03, 0.05), 3
                )
            cancel_rate = float(np.clip(cancel_rate, 0.0, 0.60))

            # ── Surge factor ──────────────────────────────────
            # Random weekly surge exposure
            surge_factor = round(random.uniform(1.1, 2.2), 2)

            # ── Earnings ──────────────────────────────────────
            earnings = compute_weekly_earnings(
                rides_this_week, vehicle,
                surge_factor, cancel_rate
            )

            # Earnings drop signal — at risk drivers often complain
            # about earnings falling before churning
            if will_churn and weeks_to_churn <= 4:
                earnings *= random.uniform(0.60, 0.85)
                earnings  = round(earnings, 2)

            # ── Rating change ─────────────────────────────────
            # Weekly rating fluctuation
            weekly_rating = float(np.clip(
                (rating or 4.0) + random.uniform(-0.15, 0.15),
                1.0, 5.0
            ))
            if rides_this_week == 0:
                weekly_rating = None

            # ── Incentive received ────────────────────────────
            incentive_received = random.random() < 0.30
            incentive_amount   = round(
                random.uniform(50, 500), 2
            ) if incentive_received else 0.0

            # Incentive removal signal — churning drivers often
            # lose eligibility or platform removes bonuses
            if will_churn and weeks_to_churn <= 2:
                incentive_received = False
                incentive_amount   = 0.0

            # ── Complaints this week ──────────────────────────
            complaints = 0
            if rides_this_week > 0:
                complaint_prob = 0.05 + (
                    0.15 if (will_churn and weeks_to_churn <= 3)
                    else 0.0
                )
                complaints = np.random.poisson(
                    complaint_prob * rides_this_week / 10
                )

            # ── Zone coverage ─────────────────────────────────
            zones_covered = min(
                dr.get("zones_known", 10),
                max(1, int(rides_this_week / 5) + 1)
            ) if rides_this_week > 0 else 0

            # ── App login days ────────────────────────────────
            # Even inactive drivers sometimes open the app
            if rides_this_week > 0:
                login_days = random.randint(
                        1, max(1, min(7, rides_this_week // 8 + 1))
                    )
            elif will_churn and weeks_to_churn <= 0:
                login_days = random.randint(0, 2)
            else:
                login_days = 0

            # ── Intentional messiness ─────────────────────────
            if random.random() < 0.02:
                earnings = None
            if random.random() < 0.015:
                cancel_rate = None
            if random.random() < 0.025:
                hours_online = None

            records.append({
                "driver_id"          : driver_id,
                "week_number"        : week,
                "week_start_date"    : week_date.strftime("%Y-%m-%d"),
                "month"              : week_date.month,
                "rides_this_week"    : rides_this_week,
                "hours_online"       : hours_online,
                "cancellation_rate"  : cancel_rate,
                "weekly_earnings"    : earnings,
                "surge_factor"       : surge_factor,
                "weekly_rating"      : weekly_rating,
                "incentive_received" : incentive_received,
                "incentive_amount"   : incentive_amount,
                "complaints"         : int(complaints),
                "zones_covered"      : zones_covered,
                "login_days"         : login_days,
                "will_churn"         : will_churn,
                "churn_reason"       : churn_reason,
            })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | "
          f"{df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  TABLE 3 — INCENTIVES
# ================================================================

def generate_incentives(drivers_df):
    """
    Incentive campaigns sent to drivers.
    Each row is one incentive offer to one driver.

    Incentive data reveals:
    - Which drivers were targeted (platform thinks they might churn)
    - Whether they responded (accepted and rode more)
    - Incentive ROI — did the incentive retain the driver?

    This is a new type of feature for this project:
    INTERVENTION history as a predictor.
    """
    print("Generating incentives table...")
    records = []

    driver_ids = (
        drivers_df.drop_duplicates(subset=["driver_id"])
                  ["driver_id"].tolist()
    )

    incentive_types = [
        "weekly_bonus",       # complete N rides, earn extra ₹X
        "peak_hour_boost",    # 1.5x earnings during peak hours
        "referral_bonus",     # bring a friend driver
        "milestone_reward",   # complete 100 total rides
        "retention_offer",    # targeted at at-risk drivers
    ]

    # Each driver gets 0-4 incentive offers during the 16 weeks
    for driver_id in driver_ids:
        n_incentives = np.random.poisson(1.5)
        n_incentives = int(np.clip(n_incentives, 0, 4))

        for j in range(n_incentives):
            offer_week = random.randint(1, 14)
            itype      = random.choice(incentive_types)

            # Offer value — retention offers are worth more
            if itype == "retention_offer":
                offer_value = round(random.uniform(300, 1000), 2)
            else:
                offer_value = round(random.uniform(50, 400), 2)

            # Was the offer accepted?
            accepted = random.random() < 0.55

            # Did it result in improved rides?
            if accepted:
                rides_increase = round(
                    random.uniform(0.05, 0.25), 2
                )
            else:
                rides_increase = 0.0

            records.append({
                "driver_id"      : driver_id,
                "offer_week"     : offer_week,
                "incentive_type" : itype,
                "offer_value"    : offer_value,
                "was_accepted"   : accepted,
                "rides_increase" : rides_increase,
            })

    df = pd.DataFrame(records)
    if len(df) > 0:
        dupes = df.sample(frac=0.005, random_state=SEED).copy()
        df    = pd.concat([df, dupes], ignore_index=True)

    print(f"  → {len(df):,} rows | "
          f"{df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  TABLE 4 — SUPPORT TICKETS
# ================================================================

def generate_support_tickets(drivers_df):
    """
    Driver support ticket history.
    Each row is one support interaction.

    Support ticket patterns predict churn:
    - Drivers who complain about earnings repeatedly are at risk
    - Unresolved tickets lead to frustration and departure
    - Multiple tickets in the pre-churn window = strong signal
    """
    print("Generating support tickets table...")
    records = []

    driver_lookup = (
        drivers_df.drop_duplicates(subset=["driver_id"])
                  .set_index("driver_id")
                  .to_dict("index")
    )

    ticket_categories = [
        "earnings_dispute",   # "I was paid less than I should be"
        "app_technical",      # "App crashed during a ride"
        "ride_cancellation",  # "User cancelled and I was penalized"
        "safety_concern",     # "Felt unsafe during a ride"
        "account_issue",      # "Can't log in / account blocked"
        "incentive_issue",    # "Bonus not credited"
        "payment_delay",      # "Payment not received on time"
        "general_query",      # General questions
    ]

    resolution_statuses = [
        "resolved", "resolved", "resolved",  # most are resolved
        "pending", "pending",
        "escalated",
        "closed_no_action",
    ]

    for driver_id, dr in driver_lookup.items():
        profile = dr.get("churn_profile", "stable")

        # Number of tickets — at-risk drivers complain more
        if profile == "stable":
            n_tickets = np.random.poisson(0.8)
        elif profile == "at_risk":
            n_tickets = np.random.poisson(2.5)
        elif profile == "high_risk":
            n_tickets = np.random.poisson(4.0)
        else:
            n_tickets = np.random.poisson(1.5)

        n_tickets = int(np.clip(n_tickets, 0, 12))

        for j in range(n_tickets):
            ticket_week = random.randint(1, 16)
            category    = random.choices(
                ticket_categories,
                weights=[0.25, 0.15, 0.15, 0.08,
                         0.12, 0.10, 0.10, 0.05]
            )[0]

            resolution = random.choice(resolution_statuses)

            # Resolution time in days
            if resolution == "resolved":
                resolution_days = random.randint(1, 5)
            elif resolution == "pending":
                resolution_days = None
            elif resolution == "escalated":
                resolution_days = random.randint(7, 21)
            else:
                resolution_days = random.randint(1, 3)

            # Driver satisfaction after ticket (1-5)
            if resolution == "resolved":
                satisfaction = random.randint(3, 5)
            elif resolution == "closed_no_action":
                satisfaction = random.randint(1, 2)
            else:
                satisfaction = random.randint(1, 3)
            if random.random() < 0.20:
                satisfaction = None

            records.append({
                "driver_id"       : driver_id,
                "ticket_week"     : ticket_week,
                "category"        : category,
                "resolution"      : resolution,
                "resolution_days" : resolution_days,
                "satisfaction"    : satisfaction,
            })

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} rows | "
          f"{df.isnull().sum().sum():,} null values")
    return df


# ================================================================
#  AGGREGATE TO DRIVER LEVEL (churn label assignment)
# ================================================================

def create_churn_labels(weekly_df):
    """
    Determine churn label for each driver based on the
    weekly activity time series.

    Churn definition:
    - Driver must have at least 1 ride in weeks 1-12 (was active)
    - Driver completed 0 rides in weeks 13-16 (went silent)
    → is_churned = 1

    Retained definition:
    - Driver was active in weeks 1-12
    - Driver completed at least 1 ride in weeks 13-16
    → is_churned = 0

    Drivers who were never active in weeks 1-12 are excluded
    (cannot churn if you were never active).
    """
    active_window = weekly_df[
        weekly_df["week_number"].between(1, 12)
    ].groupby("driver_id")["rides_this_week"].sum().reset_index()
    active_window.columns = ["driver_id", "total_rides_active_window"]

    obs_window = weekly_df[
        weekly_df["week_number"].between(13, 16)
    ].groupby("driver_id")["rides_this_week"].sum().reset_index()
    obs_window.columns = ["driver_id", "total_rides_obs_window"]

    merged = active_window.merge(obs_window, on="driver_id")

    # Only keep drivers who were active in weeks 1-12
    merged = merged[merged["total_rides_active_window"] > 0]

    # Churn = was active, now silent
    merged["is_churned"] = (
        merged["total_rides_obs_window"] == 0
    ).astype(int)

    return merged[["driver_id", "is_churned"]]


# ================================================================
#  MAIN
# ================================================================

def main():
    print("=" * 65)
    print(" RealWorld-DS-ML | Ride Sharing | Driver Churn Prediction")
    print(" City           : Bangalore")
    print(" Period         : Jan-Apr 2025 (16 weeks)")
    print(" Active window  : Weeks 1-12  (Jan-Mar)")
    print(" Observation    : Weeks 13-16 (Apr)")
    print(" Problem Type   : Binary Classification")
    print(" Target         : is_churned (1=churned, 0=retained)")
    print("=" * 65)
    print(f" Drivers        : {N_DRIVERS:,}")
    print(f" Output dir     : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 65)

    # Generate all tables
    drivers_df  = generate_drivers()
    weekly_df   = generate_weekly_activity(drivers_df)
    incentive_df= generate_incentives(drivers_df)
    tickets_df  = generate_support_tickets(drivers_df)

    # Assign churn labels
    churn_labels = create_churn_labels(weekly_df)
    drivers_df   = drivers_df.drop_duplicates(
        subset=["driver_id"]
    ).merge(churn_labels, on="driver_id", how="inner")

    # Save
    drivers_df.to_csv(
        os.path.join(OUTPUT_DIR, "drivers.csv"), index=False
    )
    weekly_df.to_csv(
        os.path.join(OUTPUT_DIR, "weekly_activity.csv"), index=False
    )
    incentive_df.to_csv(
        os.path.join(OUTPUT_DIR, "incentives.csv"), index=False
    )
    tickets_df.to_csv(
        os.path.join(OUTPUT_DIR, "support_tickets.csv"), index=False
    )

    print()
    print("=" * 65)
    print(" ✅ All tables saved to data/raw/")
    print("=" * 65)
    print(f"  drivers.csv          : {len(drivers_df):>7,} rows")
    print(f"  weekly_activity.csv  : {len(weekly_df):>7,} rows")
    print(f"  incentives.csv       : {len(incentive_df):>7,} rows")
    print(f"  support_tickets.csv  : {len(tickets_df):>7,} rows")
    print("=" * 65)

    # ── Churn rate summary ────────────────────────────────────
    churn_rate  = drivers_df["is_churned"].mean() * 100
    churn_count = drivers_df["is_churned"].sum()
    retain_count= len(drivers_df) - churn_count

    print()
    print(" Target Variable — is_churned:")
    print(f"   Churned  (1): {churn_count:>6,}  ({churn_rate:.2f}%)")
    print(f"   Retained (0): {retain_count:>6,}  "
          f"({100-churn_rate:.2f}%)")
    print(f"   Imbalance ratio: {retain_count/churn_count:.1f}:1")

    # ── Churn by profile ──────────────────────────────────────
    print()
    print(" Churn Rate by Profile:")
    for profile in CHURN_PROFILES:
        subset = drivers_df[
            drivers_df["churn_profile"] == profile
        ]
        if len(subset) > 0:
            rate = subset["is_churned"].mean() * 100
            print(f"   {profile:<18}: {rate:.1f}%  "
                  f"({len(subset):,} drivers)")

    # ── Churn by vehicle type ─────────────────────────────────
    print()
    print(" Churn Rate by Vehicle Type:")
    for vtype in VEHICLE_TYPES:
        subset = drivers_df[
            drivers_df["vehicle_type"] == vtype
        ]
        if len(subset) > 0:
            rate = subset["is_churned"].mean() * 100
            print(f"   {vtype:<8}: {rate:.1f}%  "
                  f"({len(subset):,} drivers)")

    print("=" * 65)
    print()
    print(" Next step → Open notebooks/01_data_generation.ipynb")
    print("=" * 65)


if __name__ == "__main__":
    main()