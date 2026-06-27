# 🚗 Bangalore Ride Cancellation Prediction

End-to-end multi-class classification project predicting ride
cancellation outcomes for Bangalore — simulating an Ola/Uber/Rapido
style platform with realistic, production-style messy data.

---

## 🎯 Problem Statement

Predict the outcome of a ride booking the moment it is made —
before the ride begins — using only information available at
booking time.

**Type:** Multi-class Classification
**Target Variable:** `ride_outcome`
- `0` = Completed
- `1` = Cancelled by Driver
- `2` = Cancelled by User

**Business Value:** Proactively identify cancellation-risk rides
so the platform can intervene — pre-assign backup drivers, send
user reassurance messages, or warn driver management systems —
before a cancellation actually happens.

---

## 📊 Dataset Overview

Simulated for **Bangalore, full year 2025**, across **25 real
Bangalore zones**, with realistic driver behavioral profiles,
user cancellation history, India 2025 public holidays, and
Bangalore-specific monsoon weather patterns.

| Table | Rows | Description |
|---|---|---|
| rides.csv | 60,300 | Core ride records with outcome labels |
| users.csv | 20,200 | User profiles with cancellation history |
| drivers.csv | 3,000 | Driver profiles with reliability profiles |
| weather.csv | 87,600 | Hourly weather across 10 key zones |

**Class Distribution (realistic imbalance):**
| Class | Count | % |
|---|---|---|
| Completed | 46,597 | 77.7% |
| Driver Cancel | 3,922 | 6.5% |
| User Cancel | 9,481 | 15.8% |

**Intentional data quality issues:** missing values, duplicates,
corrupt GPS readings, zero-variance columns, inconsistent formats.

---

## 🏗️ Project Workflow

| Notebook | Purpose |
|---|---|
| `01_data_generation.ipynb` | Schema understanding, class distribution analysis, structural null verification |
| `02_data_cleaning.ipynb` | Structural vs quality null separation, data leakage rule enforcement, type fixes |
| `03_eda.ipynb` | 10 business-question charts — class separation analysis per feature |
| `04_feature_engineering.ipynb` | Encoding, 7 new engineered features, class imbalance preparation |
| `05_model_training.ipynb` | 6 models × 2 imbalance strategies = 12 training runs, cross validation |
| `06_model_evaluation.ipynb` | Multi-class SHAP, segment F1 analysis, threshold tuning, business recommendations |

---

## 🔑 Key Concept: Data Leakage Prevention

The most important design decision in this project.

We predict cancellation AT BOOKING TIME — before the ride starts.
This means we can ONLY use features available at that moment.

**Excluded columns (post-outcome — would leak the answer):**
`distance_km`, `duration_min`, `fare_amount`,
`time_to_cancellation_min`, `cancelled_by`, `cancellation_reason`

**Allowed features (available at booking time):**
Driver distance, driver rating, driver acceptance rate,
user rating, user cancellation history, estimated wait time,
surge at booking, weather, zone, time features.

---

## 🔑 Key EDA Findings

| Feature | Finding | Business Meaning |
|---|---|---|
| estimated_wait_time_min | Completed: 9.36 min vs Cancelled: 13+ min | Long waits trigger cancellations |
| driver_distance_to_pickup_km | Completed: 2.19 km vs Cancelled: ~3 km | Far drivers cancel more |
| surge_at_booking | User Cancel: 1.443x vs Completed: 1.354x | High surge triggers user cancellation |
| user_cancellations_last_30d | Cancel rides: 2.528 vs Completed: 1.814 | Past behavior predicts future behavior |
| All features max correlation | 0.181 | Signal lives in combinations, not individual features |

---

## 🛠️ Feature Engineering Highlights

| Feature | Encoding | Why |
|---|---|---|
| hour, month | Sine/Cosine cyclical | Hour 23 and 0 are adjacent |
| weather_condition | Ordinal (0-5) | Natural severity order |
| pickup_zone, drop_zone | Target (cancellation rate) | Encodes zone-level cancel risk directly |
| day_of_week, vehicle_type | One-hot | No natural order |
| `is_far_driver` | New binary | Driver > 4km = higher cancel risk |
| `is_long_wait` | New binary | Wait > 12 min = strongest predictor |
| `is_high_surge` | New binary | Surge > 2.0 = user cancel trigger |
| `is_habitual_canceller` | New binary | User cancelled 3+ times last 30d |
| `combined_risk_score` | New composite (0-4) | Sum of all 4 risk flags |
| `driver_user_rating_gap` | New ratio | Friction signal between driver and user |
| `wait_per_km` | New ratio | Efficiency — time per distance unit |

**Total features: 35**

---

## ⚖️ Class Imbalance Strategy

Two strategies compared:

**Strategy A — Class Weights:**
Penalize minority class mistakes more heavily during training.
Driver Cancel mistakes cost 11.9x more than Completed mistakes.

**Strategy B — SMOTE:**
Create synthetic minority class examples to balance training data.
After SMOTE, all 3 classes have equal representation in training.

---

## 🤖 Model Comparison Results

### Strategy A — Class Weights
| Model | Macro F1 | Driver Cancel F1 | User Cancel F1 |
|---|---|---|---|
| XGBoost | 0.3901 | 0.1395 | 0.3238 |
| LightGBM | 0.3880 | 0.1551 | 0.3173 |
| Random Forest | 0.3863 | 0.1663 | 0.3200 |
| Ridge Classifier | 0.3772 | 0.1674 | 0.3062 |
| Logistic Regression | 0.3752 | 0.1639 | 0.3066 |
| Decision Tree | 0.3508 | 0.1543 | 0.2944 |

### Strategy B — SMOTE
| Model | Macro F1 | Driver Cancel F1 | User Cancel F1 |
|---|---|---|---|
| **Random Forest** | **0.3901** | **0.1189** | **0.2697** |
| Ridge Classifier | 0.3884 | 0.1474 | 0.2479 |
| Logistic Regression | 0.3782 | 0.1361 | 0.2617 |
| Decision Tree | 0.3381 | 0.1054 | 0.2637 |
| LightGBM | 0.3081 | 0.0076 | 0.0442 |
| XGBoost | 0.3076 | 0.0025 | 0.0486 |

### 5-Fold Cross Validation — True Winner
| Model | Strategy | CV Macro F1 | CV Std |
|---|---|---|---|
| **Random Forest** | **SMOTE** | **0.6353** | **±0.0025** |
| Ridge Classifier | SMOTE | 0.5338 | ±0.0020 |
| XGBoost | Class Weights | 0.3058 | ±0.0017 |

**Winner: Random Forest + SMOTE (CV Macro F1: 0.6353)**

The single 80/20 split showed 8 models nearly tied at 0.37-0.39.
Cross validation revealed Random Forest + SMOTE is genuinely
superior — 2x better than XGBoost + Class Weights in reality.

---

## 🔍 Model Evaluation — Key Findings

### Probability Uncertainty
| Actual Class | P(Completed) | P(Driver Cancel) | P(User Cancel) |
|---|---|---|---|
| Completed | 0.464 | 0.254 | 0.282 |
| Driver Cancel | 0.418 | 0.279 | 0.304 |
| User Cancel | 0.414 | 0.262 | 0.324 |

Probabilities are nearly identical across all actual classes —
confirming the model is uncertain. This is a data limitation,
not a model limitation. Booking-time features cannot fully
distinguish which class a ride belongs to.

### AUC-ROC per Class
| Class | AUC-ROC |
|---|---|
| User Cancel | 0.6351 |
| Completed | 0.6008 |
| Driver Cancel | 0.5783 |

Driver Cancel (0.578) is barely above random (0.500) —
the rarest class with weakest booking-time signals.

### Threshold Tuning — Driver Cancel
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| Default | 0.099 | 0.148 | 0.119 |
| Optimal (0.297) | 0.085 | 0.429 | 0.142 |

Lowering threshold to 0.297 nearly triples Driver Cancel recall —
catching 43% of real driver cancellations vs 15% at default.
Recommended for silent background interventions only.

### Segment Analysis
- **Driver Cancel F1 = 0 at hours 6, 12, 13** — model cannot
  identify driver cancellations during midday at all
- **Driver Cancel F1 drops in Storm weather (0.071)** — bad
  weather creates prediction uncertainty across all classes
- **User Cancel F1 improves in Storm (0.380)** — high surge +
  long wait + bad weather = learnable user cancel pattern

---

## 💡 Business Recommendations

**Immediate actions with current model:**
- Deploy User Cancel prediction for silent reassurance messages
  ("your driver is X min away") when model flags user cancel risk
- Pre-assign backup drivers silently for Driver Cancel risk rides
  at 0.297 threshold

**What would improve the model most:**
- Driver GPS movement data post-assignment (most impactful)
- Separate binary models: "cancelled at all?" then "driver or user?"
- Monthly retraining to capture seasonal behavioral shifts

**Fundamental limitation acknowledged:**
Driver cancellation decisions happen AFTER booking — when drivers
see exact user location, nearby traffic, or receive better ride
requests. None of these post-booking signals are available at
prediction time, creating an inherent ceiling on Driver Cancel
prediction quality with booking-time features alone.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` ·
`Scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` ·
`imbalanced-learn` · `Faker`

---

## 🚀 How to Run

```bash
# Install dependencies
pip install imbalanced-learn shap

# From this folder
python src/data_generator.py          # generates raw data
jupyter notebook                       # run notebooks 01-06 in order
```

---

## 📁 Folder Structure

```
cancellation_prediction/
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

## 📌 Key Learnings vs Surge Pricing Project

| Concept | Surge Pricing | Cancellation Prediction |
|---|---|---|
| Problem type | Regression | Multi-class Classification |
| Target | Continuous (1.0-4.5) | 3 categories (0/1/2) |
| Primary metric | RMSE, R² | Macro F1, per-class F1 |
| Null handling | All nulls = fix | Structural vs quality nulls |
| Data leakage risk | None | Critical — enforced strictly |
| Class imbalance | Not applicable | Class weights vs SMOTE |
| SHAP output | 1 value per feature | 3 values per feature (per class) |
| Threshold tuning | Not applicable | Precision-recall tradeoff |
| Key lesson | EDA drives model choice | Cross validation reveals true winner |