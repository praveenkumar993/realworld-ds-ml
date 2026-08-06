# 🧠 RealWorld-DS-ML

> A structured, domain-by-domain Data Science and Machine Learning portfolio built on **realistic, production-style data** — not clean Kaggle datasets.

---

## 👤 About This Repository

This repository is my personal **hands-on learning and mastery system** for Data Science and Machine Learning.

Every project here simulates how data actually exists in real companies — messy, incomplete, inconsistent, and full of edge cases. The goal is not just to train models, but to build the intuition and decision-making ability of a real DS/ML engineer working in production environments.

This work is driven by one core belief:

> **Anyone can run a model on clean data. The real skill is knowing what to do when the data is broken — and being able to explain every decision you made.**

---

## 🎯 Purpose & Goals

- Build **deep hands-on expertise** in classical ML — preprocessing, EDA, feature engineering, model selection, and evaluation
- Simulate **real production data environments** — the kind of messy, multi-table, inconsistent data you encounter at companies like Swiggy, Ola, Flipkart, and Apollo Hospitals
- Develop strong **storytelling and reasoning skills** — not just what the model outputs, but *why* certain decisions were made at every step
- Cover **multiple business domains** to understand how DS/ML problems differ by industry
- Build a **credible, well-documented portfolio** that demonstrates practical expertise to recruiters and engineering teams

---

## 🗂️ Repository Structure

```
realworld-ds-ml/
│
├── ride_sharing/          # Ola / Uber / Rapido style problems
├── healthcare/            # Patient data, diagnostics, risk prediction
├── ecommerce/             # Flipkart / Amazon style problems
├── finance/               # Fraud detection, risk scoring, forecasting
│
└── README.md              # We are here
```

> **Note:** Each domain folder will be populated progressively as work is completed. Every subfolder represents one complete end-to-end project — from raw data generation to final model evaluation.

---

## 🏗️ How Each Project Is Built

Every project in this repository follows the same rigorous, production-style workflow:

### Step 1 — Data Generation & Simulation
Raw data is generated using Python (`Faker`, `NumPy`, `random`) to simulate the kind of multi-table, messy data found in real production systems. This includes:
- Intentional missing values and nulls
- Duplicate records
- Outliers and data entry errors
- Inconsistent formats (dates, categories, units)
- Multiple related tables that need to be joined

### Step 2 — Data Cleaning & Preprocessing
This is treated as the most critical and time-intensive step — because in real jobs, it is. Every cleaning decision is documented and justified:
- Why certain rows were dropped vs imputed
- Which imputation strategy was chosen and why
- How outliers were handled (removed, capped, transformed)
- How data types were corrected and standardized

### Step 3 — Exploratory Data Analysis (EDA)
Deep visual and statistical exploration using `Pandas`, `Matplotlib`, and `Seaborn`:
- Univariate and bivariate analysis
- Distribution analysis
- Correlation heatmaps
- Business insight extraction from patterns

### Step 4 — Feature Engineering
Creating meaningful features that improve model performance:
- Domain-specific feature creation
- Encoding strategies for categorical variables
- Scaling and normalization decisions
- Feature selection and importance analysis

### Step 5 — Model Training
Multiple models trained and compared on every problem:
- Baseline model (Linear/Logistic Regression)
- Tree-based models (Decision Tree, Random Forest)
- Gradient Boosting models (XGBoost, LightGBM)
- Each model trained with proper cross-validation

### Step 6 — Model Evaluation & Selection
Every model is evaluated across multiple metrics — not just accuracy:
- Regression: RMSE, MAE, R², MAPE
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- The **best model is chosen with a written justification** explaining exactly why it outperforms the others for this specific problem and dataset

---

## 📁 Domains Covered

### 🚗 Ride Sharing  *(In Progress)*
Simulating data from a platform like Ola, Uber, or Rapido.

| Project | Problem Type | Status |
|---|---|---|
| Surge Pricing Prediction | Regression | ✅ Completed |
| Driver Churn Prediction | Classification | ✅ Completed |
| Ride Cancellation Prediction | Classification | ✅ Completed |
| ETA Prediction | Regression | ✅ Completed |
| Fraud Detection | Anomaly Detection | ✅ Completed |

---

### 🏥 Healthcare  *(Upcoming)*
Simulating patient records, diagnostic data, and clinical outcomes.

| Project | Problem Type | Status |
|---|---|---|
| Patient Readmission Prediction | Classification | 🔜 Upcoming |
| Disease Risk Scoring | Classification | 🔜 Upcoming |
| Medical Cost Prediction | Regression | 🔜 Upcoming |

---

### 🛒 E-Commerce  *(Upcoming)*
Simulating order, user, and product data from a platform like Flipkart or Amazon.

| Project | Problem Type | Status |
|---|---|---|
| Customer Churn Prediction | Classification | 🔜 Upcoming |
| Product Return Prediction | Classification | 🔜 Upcoming |
| Sales Forecasting | Time Series | 🔜 Upcoming |
| Customer Segmentation | Clustering | 🔜 Upcoming |

---

### 💰 Finance  *(Upcoming)*
Simulating transaction and customer data for banking and fintech.

| Project | Problem Type | Status |
|---|---|---|
| Credit Risk Scoring | Classification | 🔜 Upcoming |
| Transaction Fraud Detection | Anomaly Detection | 🔜 Upcoming |
| Loan Default Prediction | Classification | 🔜 Upcoming |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Manipulation | `Pandas`, `NumPy` |
| Visualization | `Matplotlib`, `Seaborn` |
| Machine Learning | `Scikit-learn`, `XGBoost`, `LightGBM` |
| Data Generation | `Faker`, `NumPy random` |
| Model Explainability | `SHAP` |
| Experiment Tracking | `MLflow` *(planned)* |
| Environment | `Jupyter Notebooks`, `Python 3.10+` |

---

## 📌 Key Principles Followed Throughout

- **No clean Kaggle data** — all datasets are synthetically generated or sourced from real APIs to mimic production environments
- **Every decision is documented** — cleaning choices, model selection, feature engineering are all explained in the notebooks
- **Depth over breadth** — one well-documented project is worth more than ten shallow ones
- **Reproducibility** — every notebook can be run end to end with a single environment setup
- **Business thinking** — every ML problem is framed in terms of real business impact, not just model metrics

---

## 📈 Progress Tracker

| Domain | Projects Completed | Total Planned |
|---|---|---|
| Ride Sharing | 5 | 5 |
| Healthcare | 0 | 3 |
| E-Commerce | 0 | 4 |
| Finance | 0 | 3 |
| **Total** | **5** | **15** |

---

## 🔗 Related Work

This repository is part of a broader portfolio that also includes:
- **Agentic AI systems** built with LangGraph, CrewAI, and MCP
- **NLP and transformer fine-tuning** projects
- **Data engineering** projects using Kafka, Spark, and Airflow


# 🧠 RealWorld-DS-ML

A production-style data science and machine learning portfolio built
with realistic, messy data simulating real business problems across
multiple domains. Every project follows a complete DS/ML workflow —
raw data generation → cleaning → EDA → feature engineering →
model training → evaluation — with detailed documentation explaining
every decision from first principles.

> **Built for:** AI Engineer / Data Scientist roles
> **Experience level modeled:** Mid-to-Senior DS/ML Engineer
> **Focus:** Classical ML depth + production thinking + business framing

---

## 🎯 What Makes This Portfolio Different

Most DS portfolios use clean Kaggle datasets with a single notebook.
This portfolio simulates what real data science looks like:

| Real-World Practice | How This Repo Does It |
|---|---|
| Messy, multi-table data | 3-4 tables per project with nulls, duplicates, outliers |
| Domain-specific knowledge | Bangalore geography, IT corridor traffic, fraud behavioral patterns |
| Physics-informed features | Haversine distance, time-speed inconsistency, expected travel time |
| Production metric framing | Beat app baseline, maximize recall at acceptable false alarm rate |
| Honest failure analysis | Stage 2 perfect scores investigated and documented as artifact |
| Business recommendations | Every evaluation ends with actionable deployment guidance |
| Multi-notebook workflow | 6-7 notebooks per project following professional DS workflow |

---

## 📁 Repository Structure

```
realworld-ds-ml/
│
├── ride_sharing/                    ← Domain 1 (Complete)
│   ├── surge_pricing/               ✅ Regression
│   ├── cancellation_prediction/     ✅ Multi-class Classification
│   ├── eta_prediction/              ✅ Regression with Baseline
│   └── fraud_detection/             ✅ Hierarchical Binary + Multi-class
│
├── healthcare/                      ← Domain 2 (Coming Soon)
│   └── ...
│
├── ecommerce/                       ← Domain 3 (Planned)
│   └── ...
│
└── README.md
```

---

## ✅ Completed Projects

### Domain 1 — Ride Sharing (Bangalore, 2025)

All four projects use the same city (Bangalore), same 25 zones
with real GPS coordinates, same public holidays, and same
monsoon weather patterns — building a coherent domain story
across four different ML problem types.

---

### 1. 🚗 Surge Pricing Prediction
**`ride_sharing/surge_pricing/`**

**Type:** Regression
**Target:** `surge_multiplier` (1.0 – 4.5)
**Data:** 5 tables, 50,250 rides

| Metric | Value |
|---|---|
| Best Model | XGBoost |
| Test RMSE | 0.1049 |
| Test R² | 0.9332 |
| Test MAPE | 4.66% |
| CV RMSE | 0.1051 ± 0.0007 |

**Key Concepts Introduced:**
- Complete DS/ML workflow from scratch
- Feature engineering for time-series behavioral patterns
- Cyclical encoding of hour and month (sin/cos)
- Target encoding of zones by average surge
- IQR outlier capping
- SHAP value analysis
- Segment-level error analysis

**Top Finding:** Non-linear hour patterns (peak hour surge 2.5x
off-peak) make tree models significantly outperform linear models.
`fare_per_km` (engineered feature) was the single strongest
predictor (importance 1.296).

---

### 2. 🚫 Ride Cancellation Prediction
**`ride_sharing/cancellation_prediction/`**

**Type:** Multi-class Classification
**Target:** `ride_outcome` (0=completed, 1=driver cancel, 2=user cancel)
**Data:** 4 tables, 60,300 rides
**Class Distribution:** 77.7% / 15.8% / 6.5%

| Metric | Value |
|---|---|
| Best Model | Random Forest + SMOTE |
| CV Macro F1 | 0.6353 ± 0.0025 |
| Primary Evaluation | Cross Validation (single split was misleading) |

**Key Concepts Introduced:**
- Multi-class classification
- Structural nulls vs quality nulls
- Data leakage prevention (booking-time features only)
- Class imbalance — SMOTE vs class weights comparison
- Macro F1 as primary metric (not accuracy)
- Threshold tuning for imbalanced classes
- Cross validation revealing true winner (all models tied on single split)
- `driver_cancellations_today` zero variance — same pattern as ETA

**Top Finding:** Single split showed all models performing identically
(F1 ≈ 0.39). Cross validation revealed Random Forest + SMOTE as
clear winner (0.6353). CV is the only reliable evaluation method
when classes are imbalanced.

---

### 3. ⏱️ Driver ETA Prediction
**`ride_sharing/eta_prediction/`**

**Type:** Regression with Real-World Baseline to Beat
**Target:** `actual_arrival_time_min` (continuous minutes)
**Data:** 4 tables, 60,300 assignments
**Baseline to Beat:** App MAE=2.09 min, RMSE=3.36 min, MAPE=12.58%

| Metric | App Baseline | XGBoost | Improvement |
|---|---|---|---|
| MAE | 2.09 min | **1.30 min** | ✅ 37.7% better |
| RMSE | 3.36 min | **1.96 min** | ✅ 41.7% better |
| MAPE | 12.58% | 21.86% | ❌ 73.7% worse |

**Key Concepts Introduced:**
- Haversine distance formula (geographic math)
- Road-to-straight ratio (route complexity)
- Log transformation of right-skewed target variable
- Multicollinearity detection and removal
- Physics-based feature engineering (expected travel time)
- Interaction features (traffic × distance)
- Zone pair target encoding (corridor-level patterns)
- MAPE paradox — winning on MAE/RMSE, losing on MAPE

**Top Finding:** `traffic_distance_interaction` (engineered)
became the strongest feature (correlation 0.895) — beating
raw `road_distance_km` (0.811). Severe traffic on a 3-4km ride
produced **9x longer ETA** than normal traffic — completely
non-linear, causing linear regression to score RMSE 7.28
(worse than the app baseline).

---

### 4. 🔍 Ride Transaction Fraud Detection
**`ride_sharing/fraud_detection/`**

**Type:** Hierarchical Binary + Multi-class Classification
**Stage 1 Target:** `is_fraud` (0/1)
**Stage 2 Target:** `fraud_label` (1=driver, 2=user, 3=collusion)
**Data:** 4 tables, 60,300 transactions
**Class Distribution:** 96.96% legitimate / 3.04% fraud (32:1)

| Metric | Stage 1 (LightGBM) |
|---|---|
| Average Precision | **0.8795** (baseline: 0.0305) |
| ROC-AUC | 0.9750 |
| F-beta (β=2) | 0.8259 |
| Recall at threshold | 81.1% |
| False alarm rate | 0.32% |
| Precision at threshold | 88.9% |

**End-to-End Pipeline (per 100,000 transactions):**

| Outcome | Count |
|---|---|
| Real fraud caught | 2,475 |
| Real fraud missed | 575 |
| False alarms | 308 |
| Investigation efficiency vs random | **30x** |

**Key Concepts Introduced:**
- Extreme class imbalance (32:1 — hardest in repo)
- Average Precision as primary metric (accuracy banned)
- F-beta score β=2 (recall weighted 2x over precision)
- Precision-Recall curve vs ROC curve
- Isolation Forest (anomaly detection)
- One-Class SVM (anomaly detection)
- Unsupervised learning on legitimate-only data
- Hierarchical two-stage model pipeline
- Missing value as fraud evidence (missing indicator encoding)
- GPS spoofing behavioral fingerprint
- Network relationship features (pair frequency)
- Velocity features (rides_today, hourly counts)
- Physics-based fraud signal (time-speed inconsistency)
- Composite fraud indicator features
- End-to-end pipeline error propagation analysis
- Perfect scores investigation (Stage 2 artifact documented)

**Top Findings:**
- `time_speed_inconsistency` (physics-based feature) was #1
  LightGBM feature — completing a 10km ride in 3 minutes
  at 140kmph is physically impossible and perfectly flags fraud
- Composite features `is_collusion_suspect` (pair_freq≥2) and
  `is_high_velocity_driver` (rides_today≥15) were both perfect
  rules — 100% catch rate, 0% false positives for their target type
- Stage 2 achieved 1.0000 Macro F1 due to data simulation artifact —
  diagnosed, documented, and explained honestly

---

## 📈 Skills & Concepts Coverage Matrix

### Core ML Concepts

| Concept | Surge | Cancel | ETA | Fraud |
|---|---|---|---|---|
| Regression | ✅ | — | ✅ | — |
| Binary Classification | — | — | — | ✅ |
| Multi-class Classification | — | ✅ | — | ✅ |
| Hierarchical Pipeline | — | — | — | ✅ |
| Anomaly Detection | — | — | — | ✅ |
| Cross Validation | ✅ | ✅ | ✅ | ✅ |
| SHAP Analysis | ✅ | ✅ | ✅ | ✅ |

### Data Engineering

| Concept | Surge | Cancel | ETA | Fraud |
|---|---|---|---|---|
| Multi-table joins | ✅ | ✅ | ✅ | ✅ |
| Structural nulls | — | ✅ | — | — |
| Quality nulls | ✅ | ✅ | ✅ | ✅ |
| Null as evidence | — | — | — | ✅ |
| Missing indicator encoding | — | — | — | ✅ |
| Outlier capping (IQR) | ✅ | ✅ | ✅ | ✅ |
| Physics-based validation | — | — | ✅ | ✅ |

### Feature Engineering

| Concept | Surge | Cancel | ETA | Fraud |
|---|---|---|---|---|
| Cyclical encoding | ✅ | ✅ | ✅ | ✅ |
| Target encoding | ✅ | ✅ | ✅ | ✅ |
| One-hot encoding | ✅ | ✅ | ✅ | ✅ |
| Ordinal encoding | — | ✅ | ✅ | ✅ |
| Physics-based features | — | — | ✅ | ✅ |
| Interaction features | — | — | ✅ | ✅ |
| Zone pair encoding | — | — | ✅ | — |
| Composite indicators | — | — | — | ✅ |
| Log transformation | — | — | ✅ | — |
| Multicollinearity handling | — | — | ✅ | — |
| Velocity features | — | — | — | ✅ |
| Network features | — | — | — | ✅ |

### Class Imbalance

| Concept | Surge | Cancel | ETA | Fraud |
|---|---|---|---|---|
| Class weights | — | ✅ | — | ✅ |
| SMOTE | — | ✅ | — | ✅ |
| Stratified splitting | — | ✅ | — | ✅ |
| Threshold tuning | — | ✅ | — | ✅ |
| F-beta scoring | — | — | — | ✅ |
| Precision-Recall curve | — | — | — | ✅ |
| Average Precision | — | — | — | ✅ |

### Evaluation & Business Framing

| Concept | Surge | Cancel | ETA | Fraud |
|---|---|---|---|---|
| Beat real-world baseline | — | — | ✅ | — |
| Segment-level analysis | ✅ | ✅ | ✅ | ✅ |
| Business cost framing | — | — | — | ✅ |
| MAPE paradox | — | — | ✅ | — |
| Perfect scores = investigate | — | — | — | ✅ |
| End-to-end pipeline eval | — | — | — | ✅ |
| Deployment recommendations | ✅ | ✅ | ✅ | ✅ |

---

## 🔢 Portfolio Statistics

| Stat | Value |
|---|---|
| Total projects | 4 |
| Total notebooks | 27 |
| Total rows of data generated | ~242,600 |
| ML problem types covered | 4 (regression, binary, multi-class, anomaly) |
| Total models trained | 40+ |
| Domains completed | 1 of 3 |

---

## 🗺️ Roadmap

### Domain 2 — Healthcare (Next)
**Planned Projects:**
- Patient Readmission Prediction (binary classification, medical data)
- Disease Risk Scoring (regression, lab values + demographics)
- Treatment Outcome Prediction (survival analysis concepts)

**New Concepts Planned:**
- Highly imbalanced medical outcomes
- Missing not at random (MNAR) — clinical data patterns
- Time-to-event features
- Calibration curves for medical probability outputs
- Cost-sensitive learning for high-stakes outcomes

### Domain 3 — E-Commerce (Planned)
**Planned Projects:**
- Customer Churn Prediction
- Product Return Prediction
- Sales Forecasting

**New Concepts Planned:**
- Time-series feature engineering
- Cohort analysis features
- Customer lifetime value
- Recommendation system fundamentals

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Core DS/ML | Python, Pandas, NumPy, Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Data Generation | Faker, custom simulation logic |
| Geographic Math | Haversine formula, road ratio encoding |
| Environment | Anaconda, Jupyter Notebooks, venv |

---

## 🚀 How to Run Any Project

```bash
# Clone the repo
git clone https://github.com/praveenkumar993/realworld-ds-ml.git
cd realworld-ds-ml

# Set up environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm shap imbalanced-learn faker

# Run any project
cd ride_sharing/surge_pricing
python src/data_generator.py
jupyter notebook
# Open notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
```

---

## 📖 How to Read This Portfolio

Each project is self-contained and can be read independently.
The recommended reading order within a domain is:
1. **README.md** — understand the business problem and results
2. **Notebook 01** — see the raw data structure
3. **Notebook 03 (EDA)** — understand the key findings
4. **Notebook 05 (Model Training)** — see the model decisions
5. **Notebook 06 (Evaluation)** — read the business recommendations

For a quick overview of skills across all projects,
the **Skills & Concepts Coverage Matrix** above shows
exactly which techniques appear in which projects.

---

## 👤 About This Portfolio

Built as a structured learning portfolio to demonstrate
classical ML competency alongside existing GenAI/MLOps
experience. Every notebook is written as if explaining
to a junior data scientist — with first-principles reasoning
behind every decision, not just code.

The goal is not to achieve the highest Kaggle score.
The goal is to demonstrate how a senior data scientist
thinks, makes decisions, documents findings honestly,
and frames results in business terms.

See my other repositories for that work.

---

## 📬 Connect

If you are a recruiter, engineer, or fellow learner who found this useful or wants to discuss any of the work here, feel free to connect.

---

*This repository is updated regularly as new projects are completed. Last updated: June 2026
