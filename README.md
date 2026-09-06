# 🧠 RealWorld-DS-ML

> A structured, domain-by-domain Data Science and Machine Learning portfolio built on **realistic, production-style data** — not clean Kaggle datasets.

---

## 👤 About This Repository

This repository is my personal **hands-on learning and mastery system** for Data Science and Machine Learning.

Every project here simulates how data actually exists in real companies — messy, incomplete, inconsistent, and full of edge cases. The goal is not just to train models, but to build the intuition and decision-making ability of a real DS/ML engineer working in production environments.

> **Anyone can run a model on clean data. The real skill is knowing what to do when the data is broken — and being able to explain every decision you made.**

---

## 🎯 Purpose & Goals

- Build **deep hands-on expertise** in classical ML — preprocessing, EDA, feature engineering, model selection, and evaluation
- Simulate **real production data environments** — the kind of messy, multi-table, inconsistent data you encounter at companies like Swiggy, Ola, Flipkart, and HDFC Bank
- Develop strong **storytelling and reasoning skills** — not just what the model outputs, but *why* certain decisions were made at every step
- Cover **multiple business domains** to understand how DS/ML problems differ by industry
- Build a **credible, well-documented portfolio** that demonstrates practical expertise to recruiters and engineering teams

---

## 🗂️ Repository Structure

```
realworld-ds-ml/
│
├── ride_sharing/                    ← Domain 1 (Complete — 5 Projects)
│   ├── surge_pricing/               ✅ Regression
│   ├── cancellation_prediction/     ✅ Multi-class Classification
│   ├── eta_prediction/              ✅ Regression with Baseline
│   ├── fraud_detection/             ✅ Hierarchical Binary + Multi-class
│   └── driver_churn/                ✅ Binary Classification + Time Series
│
├── finance/                         ← Domain 2 (In Progress)
│   └── personal_finance_clv/        ✅ Regression + PySpark + SQLite
│
├── healthcare/                      ← Domain 3 (Coming Soon)
│
└── README.md
```

---

## 🏗️ How Each Project Is Built

Every project follows the same rigorous, production-style workflow:

**Step 1 — Data Generation:** Raw data generated using Python to simulate messy, multi-table production data including intentional nulls, duplicates, outliers, and inconsistencies.

**Step 2 — Data Cleaning:** Every cleaning decision is documented and justified — why rows were dropped vs imputed, which strategy was chosen, how outliers were handled.

**Step 3 — EDA:** Deep visual and statistical exploration to extract business insights before touching a model.

**Step 4 — Feature Engineering:** Domain-specific feature creation, encoding strategies, scaling decisions, and feature selection.

**Step 5 — Model Training:** Multiple models trained and compared with proper cross-validation.

**Step 6 — Evaluation:** Every model evaluated across multiple metrics with written justification and business recommendations.

---

## ✅ Completed Projects

---

### Domain 1 — Ride Sharing (Bangalore, 2025)

All five projects share the same city (Bangalore), 25 zones with real GPS coordinates, same public holidays, and same monsoon weather patterns — a coherent domain story across five different ML problem types.

---

### 1. 🚗 Surge Pricing Prediction
**`ride_sharing/surge_pricing/`**

**Type:** Regression | **Target:** `surge_multiplier` (1.0–4.5)

| Metric | Value |
|---|---|
| Best Model | XGBoost |
| Test RMSE | 0.1049 |
| Test R² | 0.9332 |
| CV RMSE | 0.1051 ± 0.0007 |

**Key Concepts:** Cyclical encoding · Target encoding · IQR capping · SHAP · Segment error analysis

**Top Finding:** `fare_per_km` (engineered) was the strongest predictor. Non-linear hour patterns make tree models outperform linear models significantly.

---

### 2. 🚫 Ride Cancellation Prediction
**`ride_sharing/cancellation_prediction/`**

**Type:** Multi-class Classification | **Target:** `ride_outcome` (0/1/2)
**Distribution:** 77.7% completed / 15.8% driver cancel / 6.5% user cancel

| Metric | Value |
|---|---|
| Best Model | Random Forest + SMOTE |
| CV Macro F1 | 0.6353 ± 0.0025 |

**Key Concepts:** Structural nulls · Leakage prevention · SMOTE vs class weights · Macro F1 · **CV revealing true winner** (all models tied at 0.39 on single split)

---

### 3. ⏱️ Driver ETA Prediction
**`ride_sharing/eta_prediction/`**

**Type:** Regression with Real-World Baseline

| Metric | App Baseline | XGBoost | Improvement |
|---|---|---|---|
| MAE | 2.09 min | **1.30 min** | ✅ 37.7% |
| RMSE | 3.36 min | **1.96 min** | ✅ 41.7% |
| MAPE | 12.58% | 21.86% | ❌ Worse |

**Key Concepts:** Haversine distance · Log transformation · Multicollinearity · Physics features · **MAPE paradox**

**Top Finding:** `traffic_distance_interaction` correlation 0.895. Severe traffic causes 9x longer ETA — completely non-linear.

---

### 4. 🔍 Ride Transaction Fraud Detection
**`ride_sharing/fraud_detection/`**

**Type:** Hierarchical Binary + Multi-class | **Imbalance:** 32:1

| Metric | Value |
|---|---|
| Stage 1 AP (LightGBM) | **0.8795** |
| Recall at threshold | 81.1% |
| False alarm rate | 0.32% |
| Investigation efficiency | **30x vs random** |

**Key Concepts:** Extreme imbalance · Average Precision · F-beta (β=2) · Isolation Forest · One-Class SVM · GPS spoofing detection · **Perfect scores = investigate** (Stage 2 artifact documented)

---

### 5. 📉 Driver Churn Prediction
**`ride_sharing/driver_churn/`**

**Type:** Binary Classification + Time Series
**Data:** 80,000 weekly rows → 5,000 driver-level features

| Metric | Simulated | Realistic Estimate |
|---|---|---|
| ROC-AUC | 1.0000* | 0.75-0.88 |
| Retention ROI | — | **13.6x** |

*Perfect scores diagnosed as definitional circularity — documented.

**Key Concepts:** Weekly time-series data · Panel-to-cross-section · Temporal leakage prevention · Trend features (linear slope) · Window comparison · RFM framework · **Second perfect score investigation**

**Top Finding:** Rising cancellation rate (cancel_slope) is the single strongest churn signal. 3-4 week pre-churn behavioral decline is the actionable intervention window.

---

### Domain 2 — Finance (In Progress)

---

### 6. 🏦 Indian Banking CLV Prediction
**`finance/personal_finance_clv/`**

**Type:** Regression | **Target:** `clv_next_12months` (₹)
**Scale:** 33+ million rows | **Engine:** PySpark 4.2.0 + SQLite

| Metric | Value |
|---|---|
| Best Model | Random Forest |
| Test RMSE | ₹57,519 |
| Test R² | **0.7582** |
| CV R² | 0.7470 ± 0.0082 |
| Baseline RMSE | ₹116,978 |
| RMSE Improvement | **50.8% over baseline** |

**CLV Distribution:** 8.5% of customers (very_high) generate 63.8% of portfolio CLV

**Key Concepts:** SQLite JDBC → PySpark · SparkSQL CTEs · Window functions on 16M rows · Log-space regression (log1p/expm1) · Product pivot (long→wide) · Rolling 3-month spend · Segment-level evaluation · RM assignment ROI

**Top Finding (SHAP):** `total_portfolio_value` dominates at 1.6284 SHAP — 23x higher than `monthly_income` (0.0705). What customers HAVE predicts CLV better than what they EARN.

**HNI R²=0.074** — most valuable segment (63.8% of CLV) performs worst. Documented honestly with roadmap to fix.

**Business Value:** Model-based RM assignment achieves 94.9% precision vs 83.9% income-based — 110 fewer wasted assignments per 1,000 customers.

---

## 📈 Skills & Concepts Coverage Matrix

### Core ML Problem Types

| Concept | Surge | Cancel | ETA | Fraud | Churn | CLV |
|---|---|---|---|---|---|---|
| Regression | ✅ | — | ✅ | — | — | ✅ |
| Binary Classification | — | — | — | ✅ | ✅ | — |
| Multi-class Classification | — | ✅ | — | ✅ | — | — |
| Hierarchical Pipeline | — | — | — | ✅ | — | — |
| Anomaly Detection | — | — | — | ✅ | — | — |
| Time Series Data | — | — | — | — | ✅ | — |
| Big Data (PySpark) | — | — | — | — | — | ✅ |

### Data Engineering

| Concept | Surge | Cancel | ETA | Fraud | Churn | CLV |
|---|---|---|---|---|---|---|
| Multi-table joins | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Structural nulls | — | ✅ | — | — | — | — |
| Conditional nulls | — | — | — | — | ✅ | — |
| Null as evidence | — | — | — | ✅ | — | — |
| Forward-fill (time series) | — | — | — | — | ✅ | — |
| Physics-based validation | — | — | ✅ | ✅ | — | — |
| Panel-to-cross-section | — | — | — | — | ✅ | ✅ |
| SQLite + JDBC | — | — | — | — | — | ✅ |
| PySpark distributed | — | — | — | — | — | ✅ |

### Feature Engineering

| Concept | Surge | Cancel | ETA | Fraud | Churn | CLV |
|---|---|---|---|---|---|---|
| Cyclical encoding | ✅ | ✅ | ✅ | ✅ | — | — |
| Target encoding | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Physics-based features | — | — | ✅ | ✅ | — | — |
| Composite indicators | — | — | — | ✅ | ✅ | — |
| Log transformation | — | — | ✅ | — | — | ✅ |
| Trend / slope features | — | — | — | — | ✅ | — |
| Window comparison | — | — | — | — | ✅ | ✅ |
| RFM framework | — | — | — | — | ✅ | ✅ |
| Rolling window (PySpark) | — | — | — | — | — | ✅ |
| Product pivot | — | — | — | — | — | ✅ |

### Evaluation & Business Framing

| Concept | Surge | Cancel | ETA | Fraud | Churn | CLV |
|---|---|---|---|---|---|---|
| SHAP analysis | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Beat real-world baseline | — | — | ✅ | — | — | — |
| Segment-level analysis | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Business cost framing | — | — | — | ✅ | ✅ | ✅ |
| Retention ROI | — | — | — | — | ✅ | — |
| RM assignment ROI | — | — | — | — | — | ✅ |
| Perfect scores = investigate | — | — | — | ✅ | ✅ | — |

---

## 🔢 Portfolio Statistics

| Stat | Value |
|---|---|
| Total projects | 6 |
| Total notebooks | 39 |
| Total rows of data generated | ~33.5 million |
| ML problem types covered | 5 (regression, binary, multi-class, anomaly, time series) |
| Total models trained | 54+ |
| Domains completed | 1 of 2 planned (finance in progress) |
| Largest dataset | 33M rows (CLV transactions) |
| Big data engine | PySpark 4.2.0 + SQLite JDBC |
| Perfect score investigations | 2 (both documented honestly) |

---

## 🗺️ Roadmap

### Domain 2 — Finance (Continuing)
**Completed:**
- ✅ Personal Finance CLV Prediction

**Planned:**
- 🔜 Loan Default Risk Scoring (Expected Loss formula, Basel III)
- 🔜 Insurance Claim Amount Prediction (Tweedie distribution)
- 🔜 Stock Portfolio Risk Analysis (VaR, market regimes)

### Domain 3 — Healthcare (Next Domain)
**Planned Projects:**
- Patient Readmission Prediction
- Disease Risk Scoring
- Treatment Outcome Prediction

**New Concepts Planned:**
- Missing not at random (MNAR)
- Calibration curves for medical probability outputs
- Cost-sensitive learning for high-stakes outcomes

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Core DS/ML | Python, Pandas, NumPy, Scikit-learn |
| Big Data | PySpark 4.2.0, SQLite, JDBC |
| Gradient Boosting | XGBoost, LightGBM |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Data Generation | Faker, custom simulation logic |
| Environment | Anaconda, Jupyter Notebooks, VS Code |

---

## 🚀 How to Run Any Project

```bash
# Clone the repo
git clone https://github.com/[username]/realworld-ds-ml.git
cd realworld-ds-ml

# Set up environment
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Mac/Linux

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm shap imbalanced-learn faker pyspark

# Ride sharing projects
cd ride_sharing/surge_pricing
python src/data_generator.py
jupyter notebook

# Finance CLV project (requires PySpark + SQLite JAR)
cd finance/personal_finance_clv
$env:CLV_DB_PATH = "D:\your\path\finance_clv.db"
python src/data_generator.py
jupyter notebook
```

---

## 💡 The Honest Portfolio Philosophy

Two projects produced perfect model scores (Fraud Stage 2 and
Driver Churn). In both cases the root cause was identified
immediately, realistic performance was estimated from industry
benchmarks, and the limitation was documented clearly.

One project produced a critical segment failure (HNI R²=0.074
in CLV prediction). Rather than hiding it, the README documents
the cause and provides a concrete improvement roadmap.

This pattern — being suspicious of your own best results and
honest about your worst — is what separates senior data scientists
from people who just report numbers.

---

## 📈 Progress Tracker

| Domain | Projects Done | Planned |
|---|---|---|
| Ride Sharing | 5 | 5 ✅ |
| Finance | 1 | 4 |
| Healthcare | 0 | 3 |
| **Total** | **6** | **12** |

---

*Last updated: 2026. Updated regularly as new projects are completed.*