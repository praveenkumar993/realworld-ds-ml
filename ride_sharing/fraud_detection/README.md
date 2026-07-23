# 🔍 Bangalore Ride Transaction Fraud Detection

End-to-end hierarchical fraud detection system simulating ride
transaction fraud on a Bangalore ride sharing platform — combining
supervised classification with anomaly detection in a two-stage
pipeline architecture.

---

## 🎯 Problem Statement

Detect and classify fraudulent ride transactions using only
information available at transaction completion time.

**Stage 1 — Binary Detection:** Is this transaction fraudulent?
**Stage 2 — Type Classification:** Which type of fraud occurred?

**Architecture:** Hierarchical — Stage 1 feeds Stage 2.
**Business Goal:** Maximize fraud caught while keeping false alarm
rate operationally manageable for an investigation team.

---

## 📊 Dataset Overview

Simulated for **Bangalore, full year 2025**, across **25 real
Bangalore zones** with realistic fraud behavioral patterns,
device fingerprints, velocity signals, and network relationship
patterns.

| Table | Rows | Description |
|---|---|---|
| transactions.csv | 60,300 | Main fact table — one row per ride transaction |
| drivers.csv | 3,000 | Driver profiles with behavioral fraud history |
| users.csv | 20,200 | User profiles with payment and account history |
| devices.csv | 22,000 | Device fingerprint signals — new in this project |

**Class Distribution (extreme imbalance):**
| Class | Count | % | Description |
|---|---|---|---|
| Legitimate | 58,464 | 96.96% | Normal completed rides |
| Driver Fraud | 724 | 1.20% | GPS spoofing, fake completions |
| User Fraud | 617 | 1.02% | Payment fraud, promo abuse |
| Collusion Fraud | 495 | 0.82% | Driver + user working together |
| **Total Fraud** | **1,836** | **3.04%** | |

**Imbalance ratio: 32:1 (legitimate to fraud)**

---

## 🆕 What Makes This Project Different

Most technically complex project in the entire repo:

| Concept | First Appearance |
|---|---|
| Device fingerprint table | This project |
| Extreme class imbalance (97/3) | This project |
| Precision-Recall AUC as primary metric | This project |
| F-beta score (β=2) — recall weighted | This project |
| Anomaly detection (Isolation Forest, One-Class SVM) | This project |
| Unsupervised learning on legitimate-only data | This project |
| Hierarchical two-stage pipeline | This project |
| GPS spoofing pattern detection | This project |
| Network/relationship fraud signals | This project |
| Missing value as fraud signal | This project |
| Physics-based fraud signal (time-speed inconsistency) | This project |
| Cost-asymmetric business framing | This project |
| End-to-end pipeline evaluation | This project |

---

## 🏗️ Project Workflow

| Notebook | Purpose | Key New Concept |
|---|---|---|
| `01_data_generation.ipynb` | Schema, class distribution, fraud signal preview | Device table, two target variables |
| `02_data_cleaning.ipynb` | Fraud-aware cleaning, missing indicator encoding | Null as fraud evidence |
| `03_eda.ipynb` | GPS fingerprint, device signature, network analysis, velocity signals | Detective mindset EDA |
| `04_feature_engineering.ipynb` | Composite fraud indicators, interaction features | Domain knowledge encoding |
| `05_model_training_stage1.ipynb` | 6 classifiers + 2 anomaly detectors + ensemble | Anomaly detection, AP metric |
| `06_model_training_stage2.ipynb` | 3-class fraud type classification on 1,831 rows | Tiny dataset, perfect separation analysis |
| `07_model_evaluation.ipynb` | End-to-end pipeline evaluation, SHAP, business scenarios | Pipeline error propagation |

---

## 🔑 Key EDA Findings

### GPS Fraud Fingerprint (Driver Fraud)

| Signal | Legitimate | Driver Fraud | Separation |
|---|---|---|---|
| claimed_speed_kmph | 20.06 | **134.42** | **6.7x higher** |
| route_deviation_score | 0.075 | **0.651** | **8.7x higher** |
| distance_moved_km | 6.000 | **0.646** | **10.7x LOWER** |
| gps_signal_strength | 0.825 | **0.977** | Unnaturally perfect |

Driver fraud leaves a four-signal GPS fingerprint that is
essentially impossible to produce legitimately.

### Device Signature (User + Collusion Fraud)

| Signal | Legitimate | User Fraud | Collusion |
|---|---|---|---|
| VPN usage | 4.1% | **28.2%** | **28.7%** |
| GPS mock detected | 1.6% | 17.3% | 16.6% |
| Multi-account device | 9.4% | **44.9%** | **45.7%** |

### Network Pattern (Collusion Fraud)

| Signal | Legitimate | Collusion | Note |
|---|---|---|---|
| driver_user_pair_frequency | ~0.001 | **16.64** | Near-impossible by chance |
| Optimal threshold | N/A | ≥ 2 | 100% catch, 0% false positive |

### Correlation with is_fraud (Top Features)

| Rank | Feature | Correlation |
|---|---|---|
| 1 | driver_rides_today | **0.595** |
| 2 | route_deviation_score | **0.570** |
| 3 | claimed_speed_kmph | **0.564** |
| 4 | driver_user_pair_frequency | **0.490** |
| 5 | gps_signal_strength | 0.171 |
| 6 | device_accounts_linked | 0.168 |

Strongest fraud correlations in any project in this repo.
The top 4 features all exceed 0.49 — far above any single
feature in surge pricing (0.31), cancellation (0.18), or
ETA classification (max 0.60 for regression).

---

## 🛠️ Feature Engineering Highlights

### Composite Fraud Indicators — Domain Knowledge Encoded

| Feature | Logic | Performance |
|---|---|---|
| `is_gps_spoofed` | speed>80 AND deviation>0.30 AND gps_signal>0.95 | Fires for **87.6%** of driver fraud, **0.00%** of legitimate |
| `is_collusion_suspect` | pair_frequency ≥ 2 | Catches **100%** of collusion, **0.00%** false positives |
| `is_high_velocity_driver` | driver_rides_today ≥ 15 | Catches **100%** of driver fraud, **0.05%** false positives |
| `fraud_risk_composite` | Sum of 3 binary flags (0-3) | Correlation **0.766** — strongest single feature |

### Interaction Features

| Feature | Formula | Purpose |
|---|---|---|
| `gps_device_interaction` | gps_mock_detected × route_deviation_score | Driver fraud double-signal |
| `vpn_multiaccount_interaction` | uses_vpn × device_accounts_linked | User fraud combination |
| `time_speed_inconsistency` | \|completion_time - implied_time\| | Physics-based fraud signal |

### Missing Indicator Encoding (New Technique)
For null columns that correlate with fraud, binary
`_was_missing` flag columns created BEFORE imputing —
preserving the information that the value was missing
as a model feature.

### Final Feature Counts
- Stage 1 modeling dataset: 60,000 rows, 53 features
- Stage 2 modeling dataset: 1,831 rows, 53 features

---

## 🤖 Stage 1 Model Results — Binary Fraud Detection

All models evaluated by **Average Precision (accuracy banned)**.

| Rank | Model | Type | AP | ROC-AUC | F-beta2 | Recall | False Alarm |
|---|---|---|---|---|---|---|---|
| 🥇 1 | **LightGBM** | Classifier | **0.8795** | 0.9750 | 0.8113 | 83.9% | 1.04% |
| 2 | XGBoost | Classifier | 0.8794 | 0.9786 | 0.8180 | 82.8% | 0.73% |
| 3 | Ensemble (LightGBM+IF) | Ensemble | 0.8708 | 0.9680 | 0.8224 | 82.0% | 0.52% |
| 4 | Logistic Regression | Classifier | 0.8646 | 0.9796 | 0.7211 | 90.7% | 4.35% |
| 5 | Ridge Classifier | Classifier | 0.8595 | 0.9825 | 0.7790 | 75.1% | 0.22% |
| 6 | Random Forest | Classifier | 0.8591 | 0.9768 | 0.7843 | 80.1% | 0.95% |
| 7 | Decision Tree | Classifier | 0.8166 | 0.9245 | 0.6773 | 85.0% | 4.48% |
| 8 | One-Class SVM | Anomaly | 0.7392 | 0.8963 | 0.6963 | 69.4% | 0.91% |
| 9 | Isolation Forest | Anomaly | 0.1879 | 0.8364 | 0.2851 | 28.4% | 2.20% |

**Baseline AP (random classifier): 0.0305**
**LightGBM is 28.8x above random baseline.**

### At Optimal Threshold (0.6994)
| Metric | Value |
|---|---|
| F-beta (β=2) | 0.8259 |
| Recall | 81.1% |
| Precision | 88.9% |
| Fraud caught | 297 / 366 |
| False alarms | 37 / 11,634 (0.32%) |

---

## 🔬 Stage 1 Feature Importance — The Surprise Finding

**EDA predicted composite features would dominate.
LightGBM disagreed.**

| Rank | Feature | Importance |
|---|---|---|
| 1 | time_speed_inconsistency | 1197 |
| 2 | route_deviation_score | 561 |
| 3 | claimed_speed_kmph | 541 |
| 4 | user_account_age_days | 440 |
| 5 | implied_completion_time | 434 |

The model extracted more value from **continuous underlying
signals** than from binary composite indicators — because
continuous features give finer-grained split information
than binary 0/1 flags. `time_speed_inconsistency` — a
physics-based feature we engineered — became the single
most important predictor in the entire model.

---

## ⚠️ Stage 2 — Perfect Scores and What They Mean

Stage 2 achieved Macro F1 = 1.0000 across all 10 models
with CV confirming zero variance.

**This is a data simulation artifact, not a real result.**

Diagnostic confirmed 13 instances of perfect feature
separation between fraud types:

| Feature | Driver Range | User Range | Collusion Range |
|---|---|---|---|
| is_high_velocity_driver | [1, 1] | [0, 0] | [0, 0] |
| is_collusion_suspect | [0, 0] | [0, 0] | [1, 1] |
| driver_rides_today | [15, 30] | [0, 1] | [0, 1] |

Our data generator encoded fraud types with perfectly
non-overlapping behavioral boundaries. The composite
features we built in Feature Engineering then perfectly
captured those boundaries as binary lookup keys.

**Realistic Stage 2 performance with real-world data:**
- Driver Fraud F1: 0.85-0.92 (strong GPS signals)
- Collusion Fraud F1: 0.78-0.88 (pair frequency with overlap)
- User Fraud F1: 0.45-0.65 (weak signals, hard class)
- Realistic Macro F1: 0.70-0.80

Documenting this honestly demonstrates more analytical
maturity than hiding it. Perfect scores should always
trigger a leakage investigation — not celebration.

---

## 📊 End-to-End Pipeline Performance

### Complete System Results (12,000 test transactions)

| Outcome | Count | Rate |
|---|---|---|
| True Positive (fraud caught) | 297 | 81.1% of fraud |
| False Negative (fraud escaped) | 69 | 18.9% of fraud |
| False Positive (false alarm) | 37 | 0.31% of legitimate |
| True Negative (correctly released) | 11,597 | 99.69% of legitimate |

### At Bangalore Scale (500,000 rides/day)

| Daily Outcome | Count |
|---|---|
| Real fraud caught and routed | ~12,375 |
| Real fraud missed | ~2,875 |
| False alarms (review queue) | ~1,540 |
| Investigation queue precision | **89%** |
| **Investigation efficiency vs random** | **30x** |

### Pipeline Error Types

**Type 1 — Escaped Fraud (18.9%):**
Sophisticated fraudsters who deliberately stay below
detection thresholds — rides_today kept at 14 (threshold:15),
speed kept at 70kmph (suspicious but not extreme), partial
GPS deviation. These are the fraud cases that require
quarterly model retraining to catch as patterns evolve.

**Type 2 — Type Misclassification (Stage 2):**
Not measurable in simulation due to perfect Stage 2.
In production, expect ~15-25% of caught fraud to be
routed to the wrong investigation team.

**Type 3 — False Alarms (0.31%):**
Legitimate edge cases — hardworking drivers with 14 rides
today, privacy-conscious users with VPNs, families sharing
devices. Score of 0.7971 (just above threshold 0.6994)
for the representative false alarm confirms these are
genuinely borderline cases.

---

## 💡 Anomaly Detection Analysis

### One-Class SVM (AP: 0.739)
Respectable performance for a model that never saw fraud.
Learns "what normal looks like" then flags deviations.
Key advantage: catches NEW fraud patterns that supervised
classifiers have never been trained on.

### Isolation Forest (AP: 0.1879)
Significantly underperformed. Root cause: binary composite
features (`is_gps_spoofed`, `is_collusion_suspect`) compress
the feature space into sparse binary clusters, breaking
Isolation Forest's tree-based isolation mechanism.
**Fix: retrain on continuous features only.**

### Ensemble (AP: 0.8708)
Combined LightGBM + Isolation Forest underperformed
LightGBM alone because Isolation Forest's weak AP (0.1879)
dragged down the ensemble.
**Fix: replace Isolation Forest with One-Class SVM in
ensemble for next iteration.**

---

## 🚀 Deployment Recommendations

### Two-Tier Investigation Queue

| Score Range | Action | Rationale |
|---|---|---|
| ≥ 0.95 | Auto-suspend + priority queue | Near-certain fraud |
| 0.699-0.95 | Flag + 24hr review queue | Borderline — verify first |
| < 0.699 | Release immediately | Below detection threshold |

### Stage 2 Routing by Fraud Type

| Fraud Type | Investigation Team | Key Actions |
|---|---|---|
| Driver Fraud | GPS Investigation | Check device for spoofing apps, verify physical location |
| User Fraud | Payment Fraud | Card issuer verification, promo abuse audit |
| Collusion Fraud | Network Fraud | Map full fraud network, ban simultaneously |

### Threshold Review Triggers
- Fraud rate rises above 4% → lower threshold to 0.65
- False alarm complaints spike → raise threshold to 0.75
- New fraud pattern detected → retrain Stage 1 immediately

---

## 📌 Known Limitations

| Limitation | Impact | Fix |
|---|---|---|
| Stage 2 perfect scores = simulation artifact | Results not representative of production | Add noise to data generator |
| Isolation Forest underperforms | Ensemble weakened | Retrain on continuous features only |
| Velocity features need real-time DB | Production latency risk | Streaming feature pipeline (Kafka + Flink) |
| 18.9% fraud escapes | Sophisticated fraud undetected | Quarterly retraining with new confirmed fraud labels |
| Weather segment incomplete | No rain evaluation | Full year production data needed |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` ·
`Scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` ·
`imbalanced-learn` · `Faker` · `Math` · `JSON`

---

## 🚀 How to Run

```bash
python src/data_generator.py          # generates raw data
jupyter notebook                       # run notebooks 01-07 in order
```

---

## 📁 Folder Structure

```
fraud_detection/
├── data/
│   ├── raw/              # generated messy data (4 tables)
│   └── processed/        # cleaned, engineered, evaluation outputs
├── models/               # saved Stage 1 and Stage 2 model pickles
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training_stage1.ipynb
│   ├── 06_model_training_stage2.ipynb
│   └── 07_model_evaluation.ipynb
├── src/
│   └── data_generator.py
└── README.md
```

---

## 📌 Key Learnings vs Previous Projects

| Concept | Surge | Cancellation | ETA | Fraud Detection |
|---|---|---|---|---|
| Problem type | Regression | Multi-class | Regression | Binary + Multi-class |
| Primary metric | RMSE | Macro F1 | RMSE vs baseline | Average Precision |
| Imbalance | N/A | 77/6.5/15 | N/A | **97/3 (32:1)** |
| Null handling | Quality | Structural + quality | Physics + quality | **Evidence + quality** |
| New feature type | Behavioral | Behavioral | Physics-based | **Domain rules + interactions** |
| Model type | Regression | Classifier | Regression | **Classifier + Anomaly** |
| Architecture | Single model | Single model | Single model | **Hierarchical pipeline** |
| Key lesson | EDA drives choice | CV reveals winner | Beat real baseline | **Perfect scores = investigate** |
