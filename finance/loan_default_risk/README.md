# 🏦 Indian Banking Loan Default Risk Scoring

End-to-end credit risk modeling project predicting loan default
probability and Expected Loss — simulating credit risk analytics
for an Indian retail bank under RBI and Basel III guidelines.

---

## 🎯 Problem Statement

Predict `is_defaulted` (0/1) and `expected_loss` (₹) for each
approved loan — using applicant profile, loan characteristics,
and repayment behavioral signals.

**Type:** Binary Classification + Regression
**Target 1:** `is_defaulted` — will this loan become NPA?
**Target 2:** `expected_loss` = PD × LGD × EAD (Basel III)
**Default Definition:** 90+ Days Past Due (RBI NPA standard)
**Business Goal:** Enable risk-based pricing, capital allocation,
and early warning detection for 27,831 approved loans.

---

## 📊 Dataset Overview

Simulated for **Indian retail banking, Jan 2021 - Dec 2024**,
with **50,000 loan applications** across Metro, Tier1, Tier2,
Tier3 cities.

| Table | Rows | Description |
|---|---|---|
| applicant_profile | 50,250 | Borrower demographics and financials |
| loan_applications | 50,000 | All applications (approved/rejected/pending) |
| repayment_history | 813,692 | Monthly payment records per approved loan |
| default_labels | 27,831 | Default outcomes and EL components |

**Application Funnel:**
| Status | Count | % |
|---|---|---|
| Approved | 27,831 | 55.7% |
| Rejected | 18,842 | 37.7% |
| Pending | 3,327 | 6.6% |

**Default Statistics:**
| Loan Type | Default Rate | Avg EL |
|---|---|---|
| business_loan | 12.0% | ₹23,602 |
| personal_loan | 8.6% | ₹3,423 |
| education_loan | 7.2% | ₹5,568 |
| vehicle_loan | 5.5% | ₹1,951 |
| home_loan | 4.2% | ₹4,369 |
| **Overall** | **7.02%** | **₹6,702** |

**Class Imbalance:** 13:1 (non-default:default)
**Total Portfolio EL:** ₹18.64 crore (₹186,441,515)

---

## 🆕 What Makes This Project Different

| Concept | First Appearance |
|---|---|
| Expected Loss formula (PD × LGD × EAD) | This project |
| Basel III regulatory framing | This project |
| KS Statistic (credit risk metric) | This project |
| Calibration analysis (reliability diagrams) | This project |
| Vintage analysis (origination cohort effects) | This project |
| DPD trajectory features | This project |
| SMA classification (SMA-0, SMA-1, SMA-2) | This project |
| FOIR (Fixed Obligation to Income Ratio) | This project |
| Reject inference problem | This project |
| Temporal leakage in repayment data | This project |

---

## 🏗️ Project Workflow

| Notebook | Purpose | Key Concept |
|---|---|---|
| `01_database_setup.ipynb` | JDBC setup, EL analysis, CIBIL validation, vintage analysis | Basel III EL formula |
| `02_data_quality.ipynb` | Structural null classification, has_credit_history flag | Three types of structural nulls |
| `03_eda.ipynb` | DPD trajectory, CIBIL-DTI heatmap, feature correlations | DPD early warning |
| `04_feature_engineering.ipynb` | DPD trajectory features, SMA counts, FOIR | Temporal feature design |
| `05_model_training.ipynb` | KS statistic, calibration, temporal leakage diagnosis | Architectural temporal issue |

---

## 🔑 Key EDA Findings

### CIBIL Score is a 6.6x Default Predictor
| CIBIL Band | Default Rate |
|---|---|
| 800+ Excellent | **2.8%** |
| 750-799 Very Good | 4.35% |
| 700-749 Good | 6.57% |
| 650-699 Fair | 10.84% |
| 600-649 Poor | 11.84% |
| <600 Very Poor | **18.51%** |

**6.6x difference** between best and worst CIBIL band.

### DTI Amplifies CIBIL Risk
| DTI Band | Default Rate |
|---|---|
| DTI <20% | **4.59%** |
| DTI 20-35% | 6.34% |
| DTI 35-50% | 9.47% |
| DTI 50-65% | **13.24%** |

**CIBIL 750+ + DTI 50%+: 9.63% default rate** — higher than
CIBIL 650-749 + DTI <30% at 5.8%. Over-leverage overrides
good credit score when DTI is extreme.

### DPD Trajectory — The Early Warning Signal
| Period | Non-Default Avg DPD | Default Avg DPD |
|---|---|---|
| Months 1-6 | **2.15** | **7.11** |
| Months 7-12 | 2.14 | 19.10 |
| Months 13-24 | 2.15 | 27.01 |

**DPD acceleration for defaulters: +24.88 per window.**
Non-defaulters show -0.011 (flat). The divergence starts
in months 1-6 — 3-4 months before NPA classification.

### Feature Correlations with is_defaulted
| Rank | Feature | Correlation | Note |
|---|---|---|---|
| 1 | max_dpd | **0.9421** | ⚠️ Temporal leakage |
| 2 | avg_dpd | **0.8606** | ⚠️ Temporal leakage |
| 3 | on_time_rate | 0.6333 | ⚠️ Full tenure |
| 4 | missed_count | 0.3640 | ⚠️ Full tenure |
| 5 | dpd_early | 0.XX | ✅ Safe — months 1-6 |
| 6 | cibil_score | 0.1535 | ✅ Safe |
| 7 | debt_to_income | 0.0984 | ✅ Safe |

---

## 🛠️ Feature Engineering Highlights

### DPD Trajectory Features (New in This Repo)
```python
# Early warning features — safe to use
dpd_early       = AVG(dpd) WHERE month_number <= 6
dpd_mid         = AVG(dpd) WHERE month_number BETWEEN 7 AND 12
dpd_acceleration = dpd_mid - dpd_early  # direction of change

# SMA classification counts
sma0_count = SUM(1) WHERE dpd BETWEEN 1  AND 30
sma1_count = SUM(1) WHERE dpd BETWEEN 31 AND 60
```

### FOIR — Fixed Obligation to Income Ratio
```python
total_obligation_ratio = (monthly_obligations + emi_amount)
                         / monthly_income
```
Standard RBI underwriting metric — total payment burden
as percentage of income after adding the new loan.

### Expected Loss Components
```python
expected_loss = pd_score × lgd × loan_amount_sanctioned
# pd_score : model-computed probability of default
# lgd      : 1 - recovery_rate (secured: 0.15-0.50,
#            unsecured: 0.55-0.90)
# EAD      : outstanding balance at default
```

---

## 🤖 Model Results

### First Run (with temporal leakage)
All 4 models: **AUC=1.0000, KS=100%** — diagnosed immediately.

### Second Run (42 clean features)
All 4 models: **AUC=0.9990-0.9996, KS=99.7%** — still inflated.

### Root Cause — Three-Level Diagnosis

**Level 1 — max_dpd perfect separator:**
```
Non-defaults: max_dpd in [0, 60]
Defaults    : max_dpd in [90, 120]
Zero overlap — max_dpd IS the NPA criterion (90 DPD)
```

**Level 2 — sma1_count and ever_missed near-separators:**
```
sma1_count non-defaults [0,1] vs defaults [1,1] — overlap=0
ever_missed non-defaults [0,1] vs defaults [1,1] — overlap=0
Every defaulted loan MUST have missed payments by definition.
```

**Level 3 — Architectural temporal issue:**
The repayment history table contains the FULL loan tenure
including months during and after default. Any aggregate
computed from this table will capture the default event.

### The Correct Production Design

```
WRONG (what we built):
  Features: repayment months 1 through tenure_end
  Label   : default occurred in months 1 through tenure_end
  Problem : features and label measure the same events

CORRECT (production design):
  Features: repayment behavior in months 1-12 only
  Label   : did default occur in months 13+ ?
  Why     : features come BEFORE the prediction horizon
```

### Realistic Performance With Correct Design
| Metric | Realistic Estimate |
|---|---|
| ROC-AUC | 0.78-0.88 |
| KS Statistic | 45-65% |
| Recall | 0.65-0.80 |
| Precision | 0.25-0.45 |

KS of 45-65% = Good to Excellent by RBI standards.

---

## 📊 Basel III Expected Loss Analysis

| Loan Type | Total EL | EL as % Portfolio | Avg LGD |
|---|---|---|---|
| business_loan | ₹9.99 crore | **53.6%** | 0.73 |
| home_loan | ₹3.68 crore | 19.7% | **0.33** |
| personal_loan | ₹2.36 crore | 12.7% | 0.73 |
| education_loan | ₹1.53 crore | 8.2% | 0.73 |
| vehicle_loan | ₹1.08 crore | 5.8% | **0.34** |

**Business loans = 15% of portfolio but 53.6% of EL** because
high PD (12%) AND high LGD (0.73) compound together.
Home loans have low LGD (0.33) because collateral (property)
enables 67% recovery after default.

**Portfolio EL ratio: 0.49%** — RBI requires capital reserves
equal to or exceeding this ratio. Accurate PD prediction
directly determines capital adequacy compliance.

---

## 🔍 Vintage Analysis

Default rates stable across 30 origination months (2021-2023):
- Range: 5.34% to 8.92% per monthly vintage
- Mean: ~7.0% (consistent with overall rate)
- No drift — underwriting quality stable

---

## 🆕 New Concept: Reject Inference

27,831 approved loans have default outcomes.
18,842 rejected applications have NO outcome data.

**The reject inference problem:** We only know whether
borrowers default if we approved their application. Did we
correctly reject the right people? We cannot know — there
is no counterfactual for rejected applicants.

In production, banks handle this with:
1. Through-the-door sampling (approve a random sample of marginals)
2. Augmentation techniques (assume rejected = defaulted)
3. Quasi-experimental methods

Our model is trained only on approved loans — it cannot
generalize to the rejected population.

---

## ⚠️ Known Limitations

| Limitation | Impact | Fix |
|---|---|---|
| Temporal leakage in repayment features | AUC inflated to 0.999 | Separate observation and prediction windows |
| Reject inference gap | Model only trained on approved loans | Through-the-door sampling |
| No macroeconomic features | Missing interest rate cycles, GDP effects | Add RBI repo rate, inflation as features |
| Flat vintage curve | No real underwriting drift | Add quality degradation to generator |
| Static CIBIL at application | CIBIL changes over time | Add bureau refresh features |

---

## 🛠️ Tech Stack

**Data Processing:** PySpark 4.2.0, SQLite (JDBC), SparkSQL

**ML:** Python, Pandas, Scikit-learn, XGBoost, LightGBM

**Credit Risk Metrics:** KS Statistic, Calibration curves,
Expected Loss (PD × LGD × EAD)

---

## 🚀 How to Run

```bash
# Generate database
$env:LOAN_DB_PATH = "path\to\loan_default.db"
python src/data_generator.py

# Run notebooks in order: 01 → 02 → 03 → 04 → 05
jupyter notebook
```

---

## 📁 Folder Structure

```
loan_default_risk/
├── data/
│   ├── loan_default.db      # SQLite database
│   └── processed/           # cleaned CSVs, model artifacts
├── jars/
│   └── sqlite-jdbc-3.45.1.0.jar
├── notebooks/
│   ├── 01_database_setup.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_training.ipynb
├── src/
│   └── data_generator.py
└── README.md
```

---

## 📌 Key Learnings vs CLV Project

| Concept | CLV | Loan Default |
|---|---|---|
| Target | Continuous ₹ | Binary 0/1 + ₹ EL |
| Key challenge | Right skew, HNI variance | Temporal leakage, reject inference |
| Primary metric | RMSE, R² | KS Statistic, AUC |
| New metric | Log-space regression | KS, Calibration, EL accuracy |
| Business framing | RM assignment ROI | Capital reserve calculation |
| Regulatory context | None | RBI NPA, Basel III |
| Perfect score issue | No | Yes — architectural temporal |
| Data structure | Transaction rows | Repayment monthly rows |