# 🕐 Bangalore Driver Arrival Time (ETA) Prediction

End-to-end regression project predicting how many minutes a driver
takes to reach a user's pickup location after assignment — simulating
an Ola/Uber/Rapido style platform for Bangalore, India.

---

## 🎯 Problem Statement

Predict `actual_arrival_time_min` — the number of minutes between
driver assignment and driver arriving at the pickup location — using
only information available at the moment of assignment.

**Type:** Regression — predicting a continuous number
**Target Variable:** `actual_arrival_time_min`
**Business Goal:** Beat the app's current ETA formula to reduce
user frustration, lower cancellation rates, and improve trust.

---

## 📊 Dataset Overview

Simulated for **Bangalore, full year 2025**, across **25 real
Bangalore zones** with real GPS coordinates, 11 specifically
encoded traffic corridor profiles, India 2025 public holidays,
and Bangalore monsoon weather patterns.

| Table | Rows | Description |
|---|---|---|
| assignments.csv | 60,300 | Main fact table — one row per driver assignment |
| drivers.csv | 3,000 | Driver profiles with speed behavior and area familiarity |
| users.csv | 20,200 | User profiles with pickup pin accuracy |
| traffic.csv | 105,120 | Hourly traffic per zone corridor |

**Target Variable Distribution:**
| ETA Bucket | Count | % |
|---|---|---|
| 0-5 minutes | 24,754 | 41.1% |
| 5-10 minutes | 8,123 | 13.5% |
| 10-15 minutes | 4,288 | 7.1% |
| 15-20 minutes | 3,326 | 5.5% |
| 20-30 minutes | 4,840 | 8.0% |
| 30+ minutes | 12,645 | 21.0% |

**Right-skewed distribution** — handled with log transformation.

---

## 🆕 What Makes This Project Different

This is the most technically rich of our three ride-sharing projects:

| Concept | First Appearance |
|---|---|
| Haversine distance formula | This project |
| Road-to-straight ratio (route complexity) | This project |
| Zone pair traffic corridor profiles | This project |
| Physics-based feature engineering | This project |
| Interaction features (traffic × distance) | This project |
| Log transformation of target variable | This project |
| Multicollinearity detection and handling | This project |
| Real-world baseline to beat (app formula) | This project |
| MAPE paradox analysis | This project |

---

## 🏗️ Project Workflow

| Notebook | Purpose | Key New Concept |
|---|---|---|
| `01_data_generation.ipynb` | Schema, target distribution, app vs actual ETA analysis | Haversine distance, road ratio |
| `02_data_cleaning.ipynb` | Physics-based validation, log transform target | Cross-column consistency check, log1p transform |
| `03_eda.ipynb` | 10 business questions, controlled traffic comparison | Controlled variable analysis, 9x traffic effect |
| `04_feature_engineering.ipynb` | 39 features including physics-based and interaction terms | Multicollinearity removal, zone pair encoding |
| `05_model_training.ipynb` | 6 models trained on log target, evaluated vs app baseline | Log-to-minutes reversal, baseline comparison |
| `06_model_evaluation.ipynb` | SHAP, segment analysis, MAPE paradox, deployment recommendations | MAPE vs MAE/RMSE tradeoff |

---

## 🔑 Key EDA Findings

| Feature | Correlation with ETA | Business Meaning |
|---|---|---|
| road_distance_km | **+0.811** | Distance is the dominant predictor |
| straight_line_distance_km | +0.800 | Dropped — multicollinear with road distance |
| traffic_multiplier | **+0.411** | Severe traffic causes 9x time increase |
| hour | +0.162 | Peak hours increase ETA via traffic |
| driver_speed_factor | -0.084 | Faster drivers arrive sooner |
| is_weekend | -0.081 | Weekends = less office traffic = faster |
| is_holiday | -0.072 | Holidays empty Bangalore roads |

**Critical EDA Finding:** The controlled traffic comparison showed
that severe traffic on a 3-4km ride (36.6 min) versus low traffic
(4.0 min) produces a **9x time difference on the same distance**.
This non-linearity directly determined model selection.

**App Baseline Established:**
| Metric | App Formula Performance |
|---|---|
| MAE | 2.09 minutes |
| RMSE | 3.36 minutes |
| MAPE | 12.58% |

---

## 🛠️ Feature Engineering Highlights

### New Physics-Based Features
| Feature | Formula | Why |
|---|---|---|
| `expected_travel_time_min` | road_distance / (free_flow_speed × speed_factor) × 60 | Theoretical baseline — model explains deviations |
| `traffic_distance_interaction` | traffic_multiplier × road_distance_km | Captures 9x non-linear traffic effect directly |
| `distance_traffic_ratio` | road_distance / traffic_multiplier | Effective distance under current congestion |

### Encoding Decisions
| Feature | Encoding | Reason |
|---|---|---|
| hour, month | Sine/Cosine cyclical | Circular — 23 and 0 are adjacent |
| weather_condition | Ordinal (0-5) | Natural severity order |
| pickup_zone, driver_zone | Target (avg ETA) | Zone-level ETA signal |
| zone_pair | Target (avg corridor ETA) | Corridor-specific knowledge — new |
| day_of_week, vehicle_type | One-hot | No natural order |

### Multicollinearity Handled
`straight_line_distance_km` (corr 0.984 with road_distance_km)
was dropped — keeping both would confuse linear models and
inflate feature importance scores artificially.

**Final feature count: 35**

---

## 🤖 Model Comparison Results

All models trained on `log_arrival_time_min`, evaluated in
original minutes using `np.expm1()` reversal.

| Rank | Model | Test RMSE | Test MAE | R² | MAPE | Beats App? |
|---|---|---|---|---|---|---|
| 🥇 1 | **XGBoost** | **1.959** | **1.301** | **0.990** | 21.86% | ✅ RMSE+MAE |
| 2 | LightGBM | 2.036 | 1.344 | 0.989 | 21.97% | ✅ RMSE+MAE |
| 3 | Random Forest | 2.484 | 1.522 | 0.983 | 23.78% | ✅ RMSE+MAE |
| 4 | Decision Tree | 3.484 | 2.082 | 0.967 | 27.42% | ❌ |
| 5 | Linear Regression | 7.282 | 4.311 | 0.856 | 54.60% | ❌ |
| 6 | Ridge Regression | 7.282 | 4.311 | 0.856 | 54.60% | ❌ |

**App Baseline:** RMSE=3.36 | MAE=2.09 | MAPE=12.58%

**Winner: XGBoost**

---

## 🔍 Final Model vs App Comparison

| Metric | App Formula | XGBoost | Improvement |
|---|---|---|---|
| MAE | 2.0891 min | **1.3007 min** | **✅ 37.7% better** |
| RMSE | 3.3608 min | **1.9590 min** | **✅ 41.7% better** |
| MAPE | 12.58% | 21.86% | ❌ 73.7% worse |

---

## ⚠️ The MAPE Paradox — Critical Honest Finding

XGBoost beats the app on MAE and RMSE but loses on MAPE.
This is not a contradiction — it reveals metric behavior:

MAPE calculates percentage error relative to actual value.
A 1.5-minute error on a 2-minute ride = 75% MAPE.
A 1.5-minute error on a 30-minute ride = 5% MAPE.

Since 41.1% of assignments are under 5 minutes, short ETAs
dominate the MAPE calculation. XGBoost is proportionally
less accurate on sub-3 minute ETAs but significantly better
on medium and long ETAs where absolute accuracy matters most.

**Use MAE and RMSE as primary deployment metrics.**
**Use MAPE only for short-ETA segment analysis.**

---

## 📊 Model Evaluation Key Findings

### Error Percentiles (XGBoost)
| Percentile | Absolute Error |
|---|---|
| P50 (median) | 0.888 min |
| P75 | 1.742 min |
| P90 | 2.917 min |
| P95 | 3.909 min |
| P99 | 6.682 min |

50% of predictions are within 0.888 minutes of actual.
90% are within 2.917 minutes.

### Beats App in Every Segment
- ✅ Every hour of the day (worst hour: 10am, RMSE 2.47 vs app 3.36)
- ✅ Every weather condition (worst: Storm, RMSE 2.52 vs app 3.36)
- ✅ Every zone pair corridor with sufficient data

### Systematic Bias
- Short ETAs (0-5 min): slight overestimate (-0.10 min)
- Medium/Long ETAs: slight underestimate (+0.18 to +0.58 min)
- Rain conditions: systematic underestimate — driver arrives later than predicted

---

## 💡 Why XGBoost Won

**vs Linear Models:** Linear regression scored RMSE 7.28 —
worse than the app baseline. The 9x traffic non-linearity
(same distance: normal=7 min vs severe=37 min) cannot be
represented by any linear equation. MAPE of 54.6% — completely
unusable.

**vs Decision Tree:** Just missed the app baseline (3.48 vs 3.36)
due to single-tree overfitting. No generalization of the
distance-traffic interaction.

**vs Random Forest:** RMSE 2.48 — beats app but trails XGBoost.
Independent tree averaging is less efficient than sequential
error correction for this structured tabular data.

**vs LightGBM:** Nearly identical (2.04 vs 1.96 RMSE). In
production, LightGBM would be preferred for its 3-5x faster
training speed with minimal accuracy cost.

### Feature Importance Confirms EDA
| Rank | Feature | Importance | EDA Predicted? |
|---|---|---|---|
| 1 | traffic_distance_interaction | **0.659** | ✅ Yes |
| 2 | expected_travel_time_min | 0.121 | ✅ Yes |
| 3 | traffic_multiplier | 0.086 | ✅ Yes |
| 4 | road_distance_km | 0.037 | ✅ Yes |

The engineered `traffic_distance_interaction` feature alone
captures 65.9% of the model's decision-making — direct
payoff of physics-informed feature engineering.

---

## 🚀 Deployment Recommendations

**Deploy XGBoost for:**
- Rides with road_distance_km > 2km
- Clear, Cloudy, Light Rain weather
- Off-peak hours (outside 9-10am, 5-7pm)

**Use app formula or blend for:**
- Sub-3 minute ETAs (XGBoost MAPE worse here)
- Storm conditions (highest bias)
- Zone pairs with fewer than 30 historical assignments

**Future improvements:**
1. Real-time traffic API (highest impact)
2. Driver GPS movement history (pre-assignment speed)
3. Separate model for 0-5 minute ETA bucket
4. Include `app_shown_eta_min` in modeling_df for aligned comparison

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` ·
`Scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Faker` · `Math`

---

## 🚀 How to Run

```bash
python src/data_generator.py          # generates raw data
jupyter notebook                       # run notebooks 01-06 in order
```

---

## 📁 Folder Structure

```
eta_prediction/
├── data/
│   ├── raw/            # generated messy data
│   └── processed/      # cleaned, engineered, evaluation outputs
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── src/
│   └── data_generator.py
└── README.md
```

---

## 📌 Key Learnings vs Previous Projects

| Concept | Surge Pricing | Cancellation | ETA Prediction |
|---|---|---|---|
| Problem type | Regression | Multi-class Classification | Regression |
| Target | Surge multiplier | 3 categories | Continuous minutes |
| Primary metric | RMSE, R² | Macro F1 | RMSE vs app baseline |
| Null handling | Quality nulls only | Structural + quality | Quality + physics validation |
| Data leakage | None | Critical — enforced | None |
| Class imbalance | N/A | SMOTE + class weights | N/A |
| New feature type | Behavioral | Behavioral | Physics-based + interaction |
| Target transform | None | None | Log transformation |
| Multicollinearity | Not needed | Not needed | Detected and handled |
| Key lesson | EDA drives model choice | CV reveals true winner | Beat a real baseline |