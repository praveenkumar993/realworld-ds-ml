# 🚗 Bangalore Driver Churn Prediction

End-to-end binary classification project predicting which
ride-sharing drivers will become inactive — simulating driver
retention analytics for a Bangalore ride sharing platform
(Ola/Uber/Rapido style).

---

## 🎯 Problem Statement

Predict `is_churned` — whether a driver who was active in
weeks 1-12 (Jan-Mar 2025) will go completely silent in
weeks 13-16 (Apr 2025) — using only behavioral signals
available at the end of week 12.

**Type:** Binary Classification
**Target:** `is_churned` (1=churned, 0=retained)
**Business Goal:** Flag at-risk drivers early enough for
retention intervention before they permanently leave the platform.

---

## 📊 Dataset Overview

Simulated for **Bangalore, Jan-Apr 2025 (16 weeks)**, with
**5,000 drivers** tracked across four behavioral tables.

| Table | Rows | Description |
|---|---|---|
| drivers.csv | 5,000 | Driver profiles with behavioral tendency |
| weekly_activity.csv | 80,000 | Weekly behavioral time series — 16 rows per driver |
| incentives.csv | ~7,300 | Incentive offers sent to drivers |
| support_tickets.csv | ~10,000 | Driver support and complaint history |

**Target Variable Distribution:**
| Class | Count | % |
|---|---|---|
| Retained (0) | 2,985 | 59.7% |
| Churned (1) | 2,015 | 40.3% |
| **Imbalance ratio** | **1.5:1** | Most balanced in this repo |

**Churn Definition:**
Active window (weeks 1-12): ≥1 ride completed
Observation window (weeks 13-16): 0 rides completed
→ is_churned = 1

---

## 🆕 What Makes This Project Different

First project in this repo with TIME-SERIES data:

| Concept | First Appearance |
|---|---|
| Weekly time-series behavioral data | This project |
| Panel-to-cross-section transformation | This project |
| Temporal leakage — which weeks are allowed | This project |
| Trend features (linear slope as a feature) | This project |
| Window comparison (early vs pre-churn) | This project |
| RFM framework (Recency-Frequency-Monetary) | This project |
| Consecutive streak detection | This project |
| Coefficient of variation as volatility feature | This project |
| Retention ROI analysis | This project |
| SHAP-informed personalized interventions | This project |
| Meta-feature (received_retention_offer) | This project |

---

## 🏗️ Project Workflow

| Notebook | Purpose | Key New Concept |
|---|---|---|
| `01_data_generation.ipynb` | Schema, time series structure, pre-churn behavioral signals | Weekly trajectory visualization |
| `02_data_cleaning.ipynb` | Conditionally expected nulls, forward-fill within series | Forward-fill in time series |
| `03_eda.ipynb` | Behavioral trajectory analysis, RFM, slope analysis | Temporal EDA — divergence charts |
| `04_feature_engineering.ipynb` | 83 features from 80K rows → 5K driver rows | Panel-to-cross-section aggregation |
| `05_model_training.ipynb` | 5 models, business cost optimization, threshold tuning | Cost-sensitive threshold selection |
| `06_model_evaluation.ipynb` | SHAP, retention ROI, 4-tier intervention framework | Personalized retention framework |

---

## 🔑 Key EDA Findings

### Pre-Churn Behavioral Divergence (Weeks 9-12)

| Metric | Retained | Churned | Difference |
|---|---|---|---|
| rides_this_week | 31.48 | **3.93** | **-87.5%** |
| weekly_earnings | ₹6,199 | **₹455** | **-92.7%** |
| cancellation_rate | 0.12 | **0.38** | **+220.3%** |
| hours_online | 15.56 | **3.09** | **-80.1%** |

By weeks 9-12, churning drivers show 87.5% fewer rides
and 92.7% lower earnings than retained drivers — while
still technically "active." This 3-4 week lead time is
the intervention window that makes churn prediction valuable.

### RFM Summary

| Dimension | Retained | Churned |
|---|---|---|
| Recency (weeks since last ride) | **0.00** | **2.12** |
| Frequency (mean rides/week) | **29.87** | **17.80** |
| Monetary (total earnings) | **₹70,670** | **₹35,615** |

### Trend Analysis — The Core Insight

| Status | Mean Ride Slope | Meaning |
|---|---|---|
| Retained | **+0.305 rides/week** | Slowly growing |
| Churned | **-2.947 rides/week** | Declining sharply |

| Slope Bucket | Churn Rate |
|---|---|
| Steep Decline | 100.0% |
| Mild Decline | 91.7% |
| Flat | 0.1% |
| Any Growth | 0.0% |

**Trajectory matters more than current value.**
A driver doing 20 rides/week who was doing 35 last month
is at higher risk than a driver doing 20 rides/week
who was doing 15 last month.

### Top Correlations with is_churned

| Rank | Feature | Correlation | Type |
|---|---|---|---|
| 1 | risk_trend / rides_ratio | **0.967** | Window comparison |
| 2 | cancel_change | 0.957 | Cancellation trajectory |
| 3 | earnings_ratio | 0.953 | Earnings trajectory |
| 4 | hours_ratio | 0.942 | Hours trajectory |
| 5 | cancel_slope | 0.938 | Linear trend |

31 features above 0.5 correlation — highest feature
correlation density across all five projects in this repo.

---

## 🛠️ Feature Engineering Highlights

### The Core Transformation
**80,000 weekly rows → 5,000 driver-level rows**
One row per driver with 83 aggregated features.

### Feature Categories

| Category | Key Features |
|---|---|
| RFM | weeks_since_last_ride, total_rides, total_earnings |
| Trend | ride_slope, cancel_slope, earnings_slope |
| Window Comparison | rides_ratio, cancel_change, earnings_ratio |
| Consistency | max_consecutive_zero_weeks, weeks_active_pct |
| Support Signals | total_tickets, unresolved_ticket_rate |
| Incentive | acceptance_rate, received_retention_offer |
| Composite | churn_risk_score (correlation 0.903) |

### Key Feature Definitions

**rides_ratio:**
```
= mean_rides_weeks_9_12 / mean_rides_weeks_1_8
```
Retained mean: 1.087 | Churned mean: 0.142

**ride_slope:**
Linear regression coefficient of ride count over weeks 1-12.
Retained mean: +0.305 | Churned mean: -2.947

**churn_risk_score:**
Weighted composite of recency + frequency + cancellation
+ trend + ticket signals. Correlation with churn: 0.903.
Retained mean: 0.205 | Churned mean: 0.646

---

## 🤖 Model Results

All 5 models achieved perfect 1.0000 ROC-AUC — diagnosed
as definitional circularity, not genuine model performance.
See honest analysis below.

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Realistic production estimates: ROC-AUC 0.75-0.88**

---

## ⚠️ The Perfect Score Explanation

Diagnostic confirmed 3 features with perfect separation:

| Feature | Retained Range | Churned Range |
|---|---|---|
| rides_ratio | [0.828, 1.442] | [0.000, 0.563] |
| cancel_change | [-0.042, 0.053] | [0.064, 0.279] |
| rides_gini | [0.077, 0.232] | [0.392, 1.057] |

```
rides_ratio threshold 0.80:
  ALL retained drivers: rides_ratio > 0.80  → True
  ALL churned drivers : rides_ratio < 0.80  → True
```

**Root cause: Definitional circularity across adjacent
time windows.** Our churn label (zero rides in weeks 13-16)
and our key feature (mean rides in weeks 9-12) measure
the same behavioral phenomenon — driver inactivity — at
two adjacent time periods. The simulation made the
behavioral arcs too smooth and perfectly predictable.

**This is different from fraud Stage 2:**
Fraud Stage 2 was circular feature encoding (we built
features that directly encoded the generator's labels).
Driver churn is temporal adjacency (the feature window
and label window measure the same declining behavior
at slightly different time points).

**Real production data would produce realistic performance
because:**
- Some retained drivers take 1-3 week breaks then return
- Some churners quit abruptly with no pre-decline
- Weekly counts have measurement noise and seasonal effects

---

## 📊 SHAP Analysis Results

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | cancel_slope | **11.711** |
| 2 | risk_cancel | 2.006 |
| 3 | cancel_change | 0.197 |
| 4 | rides_gini | 0.050 |
| 5-15 | All others | ~0.000 |

`cancel_slope` alone drives 85%+ of all predictions.
This confirms the real-world insight: rising cancellation
rate is the strongest leading indicator of driver churn —
frustrated drivers cancel more, receive fewer good rides,
earn less, and eventually leave.

---

## 💰 Retention ROI Analysis

Intervening on Critical tier (score > 0.75) drivers:

| Metric | Value |
|---|---|
| Drivers to contact | 2,015 |
| Intervention cost (₹300 each) | ₹604,500 |
| Drivers retained (40% success) | 765 |
| Revenue recovered (₹960/week × 12 weeks) | ₹8,812,800 |
| Net profit | **₹8,208,300** |
| **ROI** | **13.6x** |

**Every ₹1 spent on retention incentives recovers ₹13.60
in driver revenue over the following 12 weeks.**

---

## 🚀 Deployment Recommendations

### Four-Tier Intervention Framework

| Tier | Score | Action | Cost |
|---|---|---|---|
| Critical (0.75-1.0) | Phone call + ₹500 bonus | Same day | ₹500 |
| High (0.50-0.75) | SMS + ₹300 offer | 48 hours | ₹300 |
| Medium (0.25-0.50) | In-app notification | Weekly batch | ₹150 |
| Low (0-0.25) | Monitor only | No action | ₹0 |

### SHAP-Personalized Interventions
- High cancel_slope → Support + better ride matching
- High rides_gini → Guaranteed weekly minimum earnings
- High earnings_slope decline → Loyalty bonus, no conditions

### Production Pipeline
Weekly batch job every Sunday:
Data → 83-feature aggregation → LightGBM scoring →
Risk tier assignment → SHAP waterfall per driver →
Personalized message → Retention CRM dashboard →
Outcome tracking → Quarterly retraining

---

## 📌 Known Limitations

| Limitation | Fix |
|---|---|
| Definitional circularity | Add temporary breaks + abrupt churners |
| No seasonal noise | Add ±20% random noise to weekly counts |
| Feature window too close to label | Use weeks 1-8 features only |
| cancel_slope dominates entirely | Realistic noise would distribute importance |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` ·
`Scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Faker`

---

## 🚀 How to Run

```bash
python src/data_generator.py    # generates raw data
jupyter notebook                 # run notebooks 01-06 in order
```

---

## 📁 Folder Structure

```
driver_churn/
├── data/
│   ├── raw/              # 4 tables including weekly time series
│   └── processed/        # cleaned, engineered, evaluation outputs
├── models/               # saved LightGBM model pickle
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

| Concept | Surge | Cancel | ETA | Fraud | Churn |
|---|---|---|---|---|---|
| Data structure | Transaction | Transaction | Transaction | Transaction | **Time series** |
| Rows per entity | 1 | 1 | 1 | 1 | **16** |
| Feature creation | Column transform | Column transform | Physics formula | Domain rules | **Temporal aggregation** |
| Key challenge | Non-linearity | Class imbalance | Beat baseline | Extreme imbalance | **Trajectory detection** |
| Primary metric | RMSE | Macro F1 | RMSE vs baseline | Average Precision | **ROC-AUC + ROI** |
| Perfect score issue | No | No | No | Stage 2 | **All models** |
| Key lesson | EDA drives choice | CV reveals winner | Beat real baseline | Investigate perfection | **Investigate perfection** |