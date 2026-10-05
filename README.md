# Credit Card Fraud Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Overview

This project builds a binary classification system to detect fraudulent credit card transactions using supervised machine learning. The dataset originates from a real-world anonymised transaction log and is heavily imbalanced, with fraudulent transactions representing fewer than 0.2 % of all records. The project walks through every stage of a typical ML workflow — from exploratory data analysis and preprocessing through model training, evaluation, threshold analysis, and model persistence — making it suitable for academic submission, portfolio demonstration, or further research.

The focus is on technically correct, reproducible, and academically defensible methodology. No web application, REST API, or database is included; the deliverable is the trained model and the analysis notebook.

---

## Objectives

1. Load and understand the Credit Card Fraud Detection dataset.
2. Perform meaningful exploratory data analysis (EDA).
3. Preprocess data **without** data leakage (split first, scale second).
4. Handle severe class imbalance using SMOTE applied exclusively on training data.
5. Train and compare four machine learning classifiers.
6. Evaluate every model using fraud-appropriate metrics (Recall, F1, PR-AUC, ROC-AUC).
7. Select the best model based on a justified metric strategy.
8. Perform limited hyperparameter tuning on the strongest candidate using cross-validation.
9. Analyse classification threshold trade-offs.
10. Persist the final trained pipeline for future use.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Observations | ~284,807 transactions |
| Fraudulent | ~492 (≈ 0.17 %) |
| Features | `Time`, `Amount`, `V1`–`V28` (anonymised PCA components) |
| Target | `Class` — 0 = Legitimate, 1 = Fraud |
| Missing values | None |
| License | Database: Open Database License (ODbL) |

> **Note:** The dataset cannot be redistributed in this repository. Download it using the Kaggle API (see setup instructions below).

---

## Technologies Used

| Category | Library / Tool |
|---|---|
| Language | Python 3.9+ |
| Notebook | Jupyter Notebook |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Imbalance handling | Imbalanced-learn (SMOTE) |
| Model persistence | Joblib |
| Dataset download | Kaggle API |

---

## Machine Learning Workflow

```
Dataset (Kaggle API)
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Train / Test Split  (stratified, 80/20)
       │
       ▼
Feature Scaling  (StandardScaler — fit on train only)
       │
       ▼
Class Imbalance Handling  (SMOTE — train set only)
       │
       ▼
┌──────────────────────────────────────────┐
│  Model 1 — Logistic Regression           │
│  Model 2 — Decision Tree                 │
│  Model 3 — Random Forest                 │
│  Model 4 — Gradient Boosting             │
└──────────────────────────────────────────┘
       │
       ▼
Evaluation  (Precision, Recall, F1, ROC-AUC, PR-AUC)
       │
       ▼
Model Comparison Table
       │
       ▼
Hyperparameter Tuning  (best candidate, CV only)
       │
       ▼
Threshold Analysis
       │
       ▼
Final Model Evaluation on held-out test set
       │
       ▼
Feature Importance / Interpretability
       │
       ▼
Save Pipeline  →  models/fraud_detection_model.pkl
```

---

## Models Used

| Model | Description |
|---|---|
| **Logistic Regression** | Linear probabilistic classifier; fast baseline; interpretable via coefficients. |
| **Decision Tree** | Non-linear rule-based classifier; fully interpretable; prone to overfitting. |
| **Random Forest** | Ensemble of decision trees via bagging; robust; provides feature importance. |
| **Gradient Boosting** | Sequential ensemble that corrects residual errors; often strong on tabular data. |

---

## Evaluation Metrics

| Metric | Why it matters for fraud detection |
|---|---|
| **Accuracy** | Misleading when classes are severely imbalanced — reported but not primary. |
| **Precision** | Of all transactions predicted as fraud, how many actually were? |
| **Recall (Sensitivity)** | Of all actual frauds, how many did the model catch? Missing a fraud is costly. |
| **F1-score** | Harmonic mean of Precision and Recall — balances both concerns. |
| **ROC-AUC** | Ranking/discrimination ability across all thresholds. |
| **PR-AUC** | Area under the Precision-Recall curve — most informative for imbalanced data. |

---

## Project Structure

```
credit-card-fraud-detection/
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb   ← main analysis notebook
│
├── data/
│   └── .gitkeep                            ← place creditcard.csv here
│
├── models/
│   └── .gitkeep                            ← trained model saved here
│
├── reports/
│   └── project_report.md                   ← academic report
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2 — Create and activate a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Kaggle API Setup

The dataset is downloaded programmatically via the Kaggle API. You must configure your Kaggle credentials **before** running the download cell in the notebook.

### Step 1 — Create a Kaggle account

If you do not already have one, register at [https://www.kaggle.com](https://www.kaggle.com).

### Step 2 — Generate an API token

1. Go to **Account → API → Create New Token**.
2. A file called `kaggle.json` is downloaded.

### Step 3 — Place the token in the correct location

**Windows**
```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```

**Linux / macOS**
```
~/.kaggle/kaggle.json
```

Then restrict permissions (Linux/macOS):
```bash
chmod 600 ~/.kaggle/kaggle.json
```

> ⚠️ **Never commit `kaggle.json` to version control.** It is listed in `.gitignore`.

---

## Running the Notebook

```bash
jupyter notebook notebooks/credit_card_fraud_detection.ipynb
```

Run cells from top to bottom. The notebook will:

1. Check whether `data/creditcard.csv` already exists.
2. If not, download it from Kaggle automatically.
3. Execute the full ML pipeline and display all results inline.

---

## Results

The table below is populated automatically by the notebook. Representative values (do not treat as fabricated ground truth — your run may vary slightly).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — | — |
| **Best (Tuned)** | — | — | — | — | — | — |

> Run the notebook to populate this table with actual values.

---

## Limitations

- **Anonymised features:** `V1`–`V28` are PCA-transformed. Direct business interpretation of individual feature importance is not possible.
- **Class imbalance:** Even after SMOTE, the model is sensitive to threshold selection; default 0.5 is often suboptimal.
- **Static dataset:** The dataset is historical. Real-world fraud patterns evolve (concept drift). The trained model is not production-ready.
- **No transaction context:** Features such as merchant category, geolocation, and cardholder history are absent.
- **Threshold trade-off:** Lowering the threshold increases recall (fewer missed frauds) at the cost of more false alarms for legitimate customers.

---

## Future Scope

- **Real-time fraud detection** with streaming architectures (Apache Kafka, Flink).
- **Cost-sensitive learning** — assign asymmetric misclassification costs.
- **Advanced ensemble methods** — XGBoost, LightGBM, stacking.
- **Anomaly detection** — Isolation Forest, Autoencoders for zero-shot fraud.
- **Concept drift detection** — monitor and retrain models as fraud patterns change.
- **Explainable AI (XAI)** — SHAP values for individual prediction explanations.
- **Graph-based fraud detection** — model transaction networks as graphs.
- **Online/incremental learning** — update models continuously without full retraining.

---

## License

This project is licensed under the [MIT License](LICENSE).

The dataset is subject to the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/) as published on Kaggle by the Machine Learning Group — ULB.
