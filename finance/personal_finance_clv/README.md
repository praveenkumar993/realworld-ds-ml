# 🏦 Indian Banking Customer Lifetime Value (CLV) Prediction

End-to-end regression project predicting the 12-month net revenue
a retail bank expects from each customer — simulating CLV analytics
for an Indian retail bank (HDFC/ICICI/Axis style).

---

## 🎯 Problem Statement

Predict `clv_next_12months` (₹) — the total net revenue a bank
expects from each customer over the next 12 months — using
behavioral signals from the previous 12 months.

**Type:** Regression (continuous monetary target)
**Target:** `clv_next_12months` in ₹ (0 to ₹900,000)
**Business Goal:** Enable data-driven RM assignment, product
targeting, and retention prioritization across 50,000 customers.

---

## 📊 Dataset Overview

Simulated for **Indian retail banking, 2023-2024**, with
**50,000 customers** across Metro, Tier1, Tier2, Tier3 cities.

| Table | Rows | Description |
|---|---|---|
| customers | 50,250 | Customer profiles with demographics and KYC |
| products | 124,865 | Product holdings per customer |
| transactions_history | 16,608,362 | 2023 transaction records |
| transactions_label | 16,608,069 | 2024 transaction records (CLV source) |
| clv_labels | 50,000 | Computed CLV target variable |

**Total data processed: 33+ million rows — genuinely PySpark scale**

**CLV Distribution (right-skewed):**
| Bucket | Customers | % | Avg CLV | Portfolio Share |
|---|---|---|---|---|
| very_high (>₹100K) | 4,260 | 8.5% | ₹334,160 | **63.8%** |
| high (₹20K-100K) | 14,189 | 28.4% | ₹46,932 | 29.8% |
| medium (₹5K-20K) | 10,091 | 20.2% | ₹11,542 | 5.2% |
| low (₹1K-5K) | 8,948 | 17.9% | ₹2,478 | 1.0% |
| very_low (<₹1K) | 12,512 | 25.0% | ₹269 | 0.15% |

**8.5% of customers generate 63.8% of portfolio CLV**

---

## 🆕 What Makes This Project Different

First project in this repo using PySpark + SQLite architecture:

| Concept | First Appearance |
|---|---|
| SQLite as data source (JDBC read) | This project |
| PySpark for data processing | This project |
| SparkSQL for feature engineering | This project |
| Window functions on 16M rows | This project |
| Indian banking domain (CLV formula) | This project |
| Log-space regression (log1p/expm1) | This project |
| Segment-level model evaluation | This project |
| Business ROI quantification | This project |

---

## 🏗️ Project Workflow

| Notebook | Purpose | Key PySpark Concept |
|---|---|---|
| `01_database_setup.ipynb` | JDBC connection, schema, SparkSQL | SparkSession, lazy evaluation |
| `02_data_quality.ipynb` | PySpark cleaning on 16M+ rows | dropDuplicates, approxQuantile |
| `03_eda.ipynb` | SparkSQL aggregations, CLV analysis | CTEs, window functions, cache |
| `04_feature_engineering.ipynb` | 16M rows → 50K customer features | rowsBetween, F.lag, pivot |
| `05_model_training.ipynb` | Ridge/RF/XGBoost/LightGBM | Log-space regression |
| `06_model_evaluation.ipynb` | Segment R², bucket accuracy, SHAP | Business value quantification |

---

## 🔑 Key EDA Findings

### Feature Correlations with CLV
| Rank | Feature | Correlation |
|---|---|---|
| 1 | monthly_income | **0.827** |
| 2 | std_txn_amount | **0.780** |
| 3 | avg_txn_amount | **0.739** |
| 4 | avg_monthly_txns | 0.485 |
| 12 | account_age_months | **0.006** (near zero) |

### Transaction Behavior by CLV Bucket
| Bucket | Txns/Year | Avg Txn Amount |
|---|---|---|
| very_high | **610** | **₹34,288** |
| very_low | **189** | ₹12,066 |

3.2x difference in transaction frequency between top and bottom.
UPI usage uniform across all buckets (47-48%) — not a CLV signal.

### City Tier CLV
Metro avg CLV ₹60,469 vs Tier3 ₹26,307 — 2.3x difference.

---

## 🛠️ Feature Engineering Highlights

### The Core PySpark Transformation
```
16,608,362 transaction rows
        ↓ PySpark groupBy + window functions
50,000 customer-level features (49 features)
        ↓ pandas + scikit-learn
Model training
```

### Key PySpark Patterns Used

**Rolling Window (new in this repo):**
```python
w = Window.partitionBy("customer_id").orderBy("txn_month")
F.sum("monthly_debit").over(w.rowsBetween(-2, 0))  # 3-month rolling
F.lag("monthly_debit", 1).over(w)                  # MoM change
```

**Product Pivot (long to wide):**
```python
MAX(CASE WHEN product_type = 'home_loan'
         AND is_active = true THEN 1 ELSE 0 END) AS has_home_loan
```

**Engineered Ratios:**
- `income_utilization` = annual_debit / (income × 12)
- `portfolio_to_income` = total_portfolio_value / monthly_income
- `spend_trend_ratio` = last_3m_avg / first_9m_avg
- `loan_to_income` = (home_loan + personal_loan) / income

---

## 🤖 Model Results

All models trained on `log(1 + CLV)` — back-transformed to ₹.

| Rank | Model | RMSE (₹) | MAE (₹) | R² | CV R² |
|---|---|---|---|---|---|
| 🥇 1 | **Random Forest** | **₹57,519** | **₹17,918** | **0.7582** | **0.7470 ± 0.0082** |
| 2 | LightGBM | ₹61,596 | ₹17,983 | 0.7227 | 0.7453 ± 0.0075 |
| 3 | XGBoost | ₹61,743 | ₹17,949 | 0.7214 | — |
| 4 | Ridge Regression | ₹97,678 | ₹27,627 | 0.3027 | — |
| — | Baseline (mean) | ₹116,978 | ₹55,278 | 0.000 | — |

**50.8% RMSE improvement over mean baseline.**

Ridge R²=0.30 confirms CLV prediction is fundamentally non-linear
despite income having 0.827 linear correlation with CLV.

---

## 📊 SHAP Feature Importance — The Key Surprise

| Rank | Feature | Mean SHAP | EDA Predicted? |
|---|---|---|---|
| 1 | total_portfolio_value | **1.6284** | ❌ EDA ranked income #1 |
| 2 | segment_mean_clv | 0.5717 | ✅ |
| 3 | has_insurance | 0.2937 | ❌ Unexpected |
| 7 | monthly_income | 0.0705 | ❌ EDA ranked this #1 |

**The model disagrees with EDA on the most important feature.**

`total_portfolio_value` dominates at 1.6284 — 23x higher than
`monthly_income` (0.0705). The model learned that what customers
HAVE (existing product holdings) predicts CLV far better than
what they EARN (income potential).

Key business insight: a customer with ₹80L home loan and ₹20L
in FDs is more valuable than a same-income customer with no
product holdings.

---

## 📈 Segment-Level Performance

| Segment | R² | Interpretation |
|---|---|---|
| salaried_entry | **0.852** | Best — simple, predictable |
| retired | 0.758 | Strong — FD/savings dominate |
| salaried_mid | 0.672 | Good |
| self_employed | 0.647 | Good — income volatility limits |
| salaried_senior | 0.490 | Moderate |
| hni | **0.074** | **Failed — extreme value variance** |
| student | -0.178 | **Failed — worse than mean baseline** |

**HNI R²=0.074 is the critical limitation.**
Most valuable segment (63.8% of CLV) performs worst.
2,465 HNI customers with CLV ₹50K-₹900K creates 18x variance
that 49 features cannot capture without granular product data.

---

## 🎯 CLV Bucket Accuracy

| Metric | Value |
|---|---|
| Exact bucket match | **79.5%** |
| Within-one-bucket | **97.3%** |
| very_high accuracy | 66.9% |
| medium accuracy | 81.2% |

97.3% within-one-bucket is the strongest operational result —
the model never tells the bank to treat an HNI as a student.

---

## 💰 Business Value — RM Assignment

| Strategy | Precision |
|---|---|
| Income-based (no model) | 83.9% |
| CLV Model-based | **94.9%** |
| Improvement | **+11 percentage points** |

110 fewer wasted RM assignments per 1,000 customers.

---

## 🚀 Deployment Recommendations

**Four-Tier Product Targeting:**
| Predicted Bucket | Action |
|---|---|
| very_high | Private banking, home loan top-up, wealth management |
| high | Mutual fund SIP, credit card upgrade, FD |
| medium | Recurring deposit, insurance cross-sell |
| low + very_low | Digital-only, UPI cashback, no RM cost |

**HNI Improvement Roadmap:**
1. Build a separate model for HNI customers only
2. Add loan tenure and FD interest rate as features
3. Add mutual fund category (equity/debt/hybrid)
4. Add transaction recency features

---

## 📌 Known Limitations

| Limitation | Impact | Fix |
|---|---|---|
| HNI R²=0.074 | Unreliable for most valuable customers | Separate HNI model |
| Student R²=-0.178 | Worse than baseline | Use rules, not ML |
| No seasonality | Missing Diwali/year-end spikes | Add seasonal patterns |
| Products lack tenure/rate | Limits loan features | Extend data generator |
| MAPE >400% | Metric unstable for zero-CLV customers | Use RMSE/MAE only |

---

## 🛠️ Tech Stack

**Data Processing:** PySpark 4.2.0, SQLite (JDBC), SparkSQL

**ML:** Python, Pandas, Scikit-learn, XGBoost, LightGBM, SHAP

**Database:** SQLite 4.8GB, local PySpark (local[2]), winutils

---

## 🚀 How to Run

```bash
# Generate database (~10 minutes for 33M rows)
$env:CLV_DB_PATH = "D:\your\path\finance_clv.db"
python src/data_generator.py

# Download SQLite JDBC JAR to jars/ folder
# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
jupyter notebook
```

---

## 📁 Folder Structure

```
personal_finance_clv/
├── data/
│   ├── finance_clv.db          # 4.8GB SQLite (external drive)
│   └── processed/              # cleaned CSVs, model artifacts
├── jars/
│   └── sqlite-jdbc-3.45.1.0.jar
├── notebooks/
│   ├── 01_database_setup.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── src/
│   └── data_generator.py
└── README.md
```

---

## 📌 Key Learnings vs Ride Sharing Projects

| Concept | Ride Sharing | CLV (This Project) |
|---|---|---|
| Data source | CSV files | **SQLite database (JDBC)** |
| Processing engine | Pandas | **PySpark** |
| Data scale | ~60K rows | **33+ million rows** |
| Feature creation | Pandas groupBy | **SparkSQL + window functions** |
| Target type | Binary/Multi-class | **Continuous regression** |
| Target transform | None | **log1p → expm1** |
| Key challenge | Imbalance, fraud | **Right skew, HNI variance** |
| Business framing | Investigation efficiency | **RM assignment ROI** |
| Perfect scores | Stage 2 fraud, churn | **No — realistic R² 0.74** |