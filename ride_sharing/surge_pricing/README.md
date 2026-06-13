# 🚗 Bangalore Surge Pricing Prediction

End-to-end machine learning project predicting ride-sharing surge multipliers for Bangalore, simulating an Ola/Uber/Rapido-style platform — built on realistic, intentionally messy production-style data rather than clean Kaggle datasets.

---

## 🎯 Problem Statement

Predict the surge multiplier (1.0x – 4.5x) for a ride based on time, weather, location, and demand signals — enabling dynamic, explainable pricing decisions.

**Type:** Regression
**Target Variable:** `surge_multiplier`

---

## 📊 Dataset Overview

Simulated for **Bangalore, full year 2025**, across **25 real Bangalore zones**, with realistic geographic coordinates, India 2025 public holidays, and Bangalore-specific monsoon weather patterns.

| Table | Rows | Description |
|---|---|---|
| rides.csv | 50,250 | Core ride transactions (target variable here) |
| users.csv | 20,200 | Customer profiles |
| drivers.csv | 3,000 | Driver profiles |
| weather.csv | 87,600 | Hourly weather across 10 key zones |
| payments.csv | 39,191 | Payment records for completed rides |

**Intentional data quality issues injected:** missing values, duplicate records, corrupt GPS/distance values, garbage ages, inconsistent formats — mimicking real production data.

---

## 🏗️ Project Workflow

| Notebook | Purpose |
|---|---|
| `01_data_generation.ipynb` | Schema understanding, null/duplicate/outlier analysis on raw data |
| `02_data_cleaning.ipynb` | Documented cleaning decisions — imputation, outlier capping, type fixes |
| `03_eda.ipynb` | 10 business-question-driven visual analyses |
| `04_feature_engineering.ipynb` | Cyclical, ordinal, target, and one-hot encoding + new domain features |
| `05_model_training.ipynb` | 6 models trained and compared with full metrics |
| `06_model_evaluation.ipynb` | SHAP analysis, segment-wise error analysis, business recommendations |

---

## 🔑 Key EDA Findings

- Mean surge across all rides: **1.72x**
- Highest surge hours: **10am (2.16x)**, **7-8pm (~2.14x)** — matches Bangalore IT office hours
- Storm weather drives surge to **2.40x** vs **1.60x** in Clear weather (+50%)
- Highest surge zone: **MG Road (1.78x)** — commercial/entertainment hub
- Holidays correlate with surge at **0.26**, weekends at **0.20**

---

## 🛠️ Feature Engineering Highlights

| Feature | Encoding | Why |
|---|---|---|
| hour, month | Sine/Cosine (cyclical) | Hour 23 and 0 are adjacent, not distant |
| weather_condition | Ordinal (0-5) | Natural severity order: Clear → Storm |
| pickup/drop zone | Target encoding | 25 categories — encode by avg surge |
| day_of_week, vehicle_type | One-hot | No natural order |
| `demand_pressure` | New composite (0-4) | Combines peak hour + rain + holiday + weekend |
| `fare_per_km` | New ratio | Isolates surge effect from trip length |

---

## 🤖 Model Comparison Results

| Rank | Model | Test RMSE | Test R² | MAPE % |
|---|---|---|---|---|
| 🥇 1 | **XGBoost** | **0.1049** | **0.9332** | **4.66%** |
| 2 | LightGBM | 0.1089 | 0.9280 | 4.94% |
| 3 | Random Forest | 0.1528 | 0.8583 | 7.03% |
| 4 | Decision Tree | 0.1629 | 0.8389 | 7.48% |
| 5 | Ridge Regression | 0.2094 | 0.7339 | 10.15% |
| 6 | Linear Regression | 0.2094 | 0.7338 | 10.15% |

**Winner: XGBoost** — explains 93.3% of variance, confirmed via 5-fold cross-validation (CV RMSE: 0.1051 ± 0.0007).

### Why XGBoost Won
- EDA showed non-linear (U-shaped) relationships — linear models fundamentally cannot capture these
- Sequential error-correction (boosting) outperforms independent tree averaging (Random Forest)
- Small overfit gap (0.0193) and tight CV variance confirm strong generalization

### Top Predictive Features (Permutation Importance)
1. `fare_per_km` (engineered) — 1.30
2. `distance_km` — 0.65
3. `demand_pressure` (engineered) — 0.53
4. `base_fare` — 0.41

Both top engineered features outrank all raw features — validating the feature engineering approach.

---

## 🔍 Model Evaluation — Key Insight

**The model is least accurate exactly when surge matters most.**

| Condition | RMSE | vs Overall |
|---|---|---|
| Clear weather | 0.0933 | -11% |
| Storm | 0.1639 | **+56%** |
| Peak hours | ~0.13 | +25-30% |

Storm rides make up only 3.7% of the dataset — the long-tail problem. The worst 5% of predictions are disproportionately concentrated in rain (49%) and peak hours (58%) — exactly the high-surge, high-stakes scenarios.

**Business recommendation:** Apply automated pricing confidently in normal conditions; use a confidence buffer or human review for storm/peak-hour combinations.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Faker`

---

## 🚀 How to Run

```bash
# From this folder
python src/data_generator.py          # generates raw data
jupyter notebook                       # then run notebooks 01-06 in order
```

---

## 📁 Folder Structure

```
surge_pricing/
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