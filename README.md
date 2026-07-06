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
| Driver Churn Prediction | Classification | 🔜 Upcoming |
| Ride Cancellation Prediction | Classification | ✅ Completed |
| ETA Prediction | Regression | ✅ Completed |
| Fraud Detection | Anomaly Detection | 🔜 Upcoming |

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
| Ride Sharing | 3 | 5 |
| Healthcare | 0 | 3 |
| E-Commerce | 0 | 4 |
| Finance | 0 | 3 |
| **Total** | **3** | **15** |

---

## 🔗 Related Work

This repository is part of a broader portfolio that also includes:
- **Agentic AI systems** built with LangGraph, CrewAI, and MCP
- **NLP and transformer fine-tuning** projects
- **Data engineering** projects using Kafka, Spark, and Airflow

See my other repositories for that work.

---

## 📬 Connect

If you are a recruiter, engineer, or fellow learner who found this useful or wants to discuss any of the work here, feel free to connect.

---

*This repository is updated regularly as new projects are completed. Last updated: June 2026
