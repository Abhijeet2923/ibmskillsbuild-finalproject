# Credit Card Fraud Detection Using Machine Learning

**Course:** Machine Learning  
**Submission Type:** Academic Project Report  
**Dataset:** Credit Card Fraud Detection — Kaggle (MLG-ULB)  
**Random Seed:** 42  

---

## Abstract

Credit card fraud represents a significant financial and operational challenge for banking institutions worldwide. The rapid growth of digital payment systems has increased the exposure of cardholders to fraudulent transactions, making automated fraud detection a critical research and engineering problem. This project applies supervised binary classification techniques to a real-world anonymised credit card transaction dataset to distinguish legitimate transactions from fraudulent ones. The dataset comprises approximately 284,807 transactions of which only 492 (≈ 0.17 %) are fraudulent, presenting a severe class imbalance problem that renders accuracy an unsuitable primary evaluation metric. The project implements four machine learning algorithms — Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting — and evaluates them using Precision, Recall, F1-score, ROC-AUC, and Precision-Recall AUC. Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively on training data to address class imbalance without introducing data leakage. A stratified train-test split, pipeline-based preprocessing, and threshold analysis are employed to ensure reproducibility and methodological correctness. The final model is selected based on its balance of Recall, F1-score, and PR-AUC, and persisted using Joblib for deployment-ready inference.

---

## 1. Introduction

The global credit card market processes billions of transactions annually, and a non-trivial fraction of these are fraudulent. According to the Nilson Report, global card fraud losses exceeded USD 32 billion in 2021 and are projected to grow. Traditional rule-based fraud detection systems, which rely on manually crafted heuristics, are increasingly insufficient due to the adaptive nature of fraudulent behaviour and the sheer volume of transactions requiring near-real-time evaluation.

Machine learning offers a data-driven alternative: models can learn complex non-linear patterns from historical transaction data and generalise to unseen transactions. Supervised learning, specifically binary classification, is well-suited to fraud detection because historical transaction labels (fraud / legitimate) are available from chargeback and dispute resolution records.

However, fraud detection datasets are characterised by extreme class imbalance — fraudulent transactions are rare events. This imbalance makes standard accuracy an uninformative metric and necessitates specialised techniques such as class weighting, undersampling, or synthetic oversampling to prevent classifiers from simply predicting the majority class for all inputs.

This project addresses these challenges systematically, building a complete, reproducible ML pipeline from raw data ingestion through model persistence.

---

## 2. Problem Statement

Given a set of numerical features derived from credit card transaction records, build a binary classifier that assigns each transaction one of two labels:

- **Class 0** — Legitimate transaction
- **Class 1** — Fraudulent transaction

The classifier must demonstrate high **Recall** on the minority (fraud) class — i.e., it must detect as many actual fraudulent transactions as possible — while maintaining acceptable **Precision** to avoid overwhelming fraud analysts with false alerts. Raw accuracy is not the optimisation target.

---

## 3. Objectives

1. Load, inspect, and understand the Credit Card Fraud Detection dataset.
2. Conduct exploratory data analysis (EDA) to characterise the dataset.
3. Apply a rigorous preprocessing pipeline that avoids data leakage.
4. Handle severe class imbalance using SMOTE on training data only.
5. Train four classification algorithms under a consistent evaluation protocol.
6. Evaluate models using fraud-appropriate metrics: Precision, Recall, F1, ROC-AUC, PR-AUC.
7. Perform limited hyperparameter tuning on the strongest candidate using stratified cross-validation.
8. Select the final model based on a justified, fraud-detection-oriented criterion.
9. Analyse the impact of probability threshold selection on model performance.
10. Provide feature importance analysis for model interpretability.
11. Persist the final trained pipeline for reproducible inference.

---

## 4. Dataset Description

### 4.1 Source

The dataset is publicly available on Kaggle:  
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

It was collected and made available by the Machine Learning Group at Université Libre de Bruxelles (ULB). The transactions occurred over two days in September 2013 by European cardholders.

### 4.2 Characteristics

| Property | Value |
|---|---|
| Total observations | 284,807 |
| Fraudulent transactions | 492 |
| Legitimate transactions | 284,315 |
| Fraud rate | ≈ 0.172 % |
| Number of features | 30 (excluding target) |
| Target column | `Class` (0 = Legitimate, 1 = Fraud) |
| Missing values | None |
| Duplicate rows | Present — addressed during EDA |

### 4.3 Features

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount in Euros |
| `V1`–`V28` | Anonymised, PCA-transformed numerical features |
| `Class` | Target label: 0 = Legitimate, 1 = Fraud |

> **Important:** `V1`–`V28` are the result of a Principal Component Analysis transformation applied to protect cardholder confidentiality. The original features are not disclosed. Consequently, direct business interpretation of these features is not possible.

### 4.4 Class Distribution

The dataset exhibits severe class imbalance. Fraudulent transactions account for approximately 0.172 % of all records. This imbalance is a fundamental characteristic of the problem — fraud is, by definition, rare. Any model that predicts "Legitimate" for every transaction achieves ~99.83 % accuracy but has 0 % Recall on fraud, making it entirely useless for the stated objective.

---

## 5. Exploratory Data Analysis

### 5.1 Dataset Structure

The dataset loads as a 284,807-row × 31-column DataFrame. All 31 columns contain numeric (float64) values. No categorical encoding is required.

### 5.2 Missing Values

A systematic `isnull().sum()` check confirms that no features contain missing values. This removes the need for imputation.

### 5.3 Duplicate Records

Duplicate rows are checked using `df.duplicated().sum()`. A small number of exact duplicates may be present. In a transaction context, duplicate rows could represent real repeated transactions (same amount, same second) or data entry artefacts. For this project, duplicates are dropped after explicit confirmation to prevent the model from training on repeated identical records, which could introduce subtle bias.

### 5.4 Class Imbalance

The class distribution is:

- **Class 0 (Legitimate):** ~284,315 transactions (~99.83 %)
- **Class 1 (Fraud):** ~492 transactions (~0.17 %)

A bar chart confirms the extreme imbalance. This imbalance motivates the use of SMOTE and the selection of PR-AUC as a primary evaluation metric.

### 5.5 Transaction Amount

- The `Amount` feature is right-skewed, with the majority of transactions below €100 and a long tail extending to several thousand Euros.
- Fraudulent transactions tend to have lower median amounts than legitimate transactions, though the distributions overlap significantly.
- `Amount` requires standardisation before modelling (StandardScaler applied post-split).

### 5.6 Time Feature

- `Time` represents seconds elapsed since the first transaction in the dataset (not clock time).
- The distribution shows two peaks, likely corresponding to daytime activity across two days.
- `Time` is retained and standardised along with `Amount`. The V features are already in a scaled PCA space and do not require further scaling, though including them in the scaler is harmless.

### 5.7 Key Observations

- No missing data — preprocessing is simplified.
- Extreme class imbalance requires dedicated handling.
- All features are numerical — no encoding required.
- PCA features prevent direct domain interpretation.
- `Amount` and `Time` are the only non-anonymised features with potential business meaning.

---

## 6. Data Preprocessing

### 6.1 Train-Test Split

The dataset is split into training (80 %) and testing (20 %) sets using a stratified split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Stratification** is critical here. Because fraud accounts for only 0.17 % of records, an unstratified random split could — by chance — place disproportionately few or many fraud cases in the test set. `stratify=y` ensures both splits maintain approximately the same class ratio as the original dataset.

**Training set size:** ~227,845 rows  
**Test set size:** ~56,962 rows  
**Fraud in test set:** ~98 cases (preserving original distribution)

### 6.2 Feature Scaling

Only `Amount` and `Time` require scaling; `V1`–`V28` are already in a normalised PCA space. `StandardScaler` is used.

**Critical methodological requirement:** The scaler is fitted **only on training data** and applied (transform only) to the test set:

```python
scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
```

Fitting the scaler on the full dataset would allow test-set statistics to influence the training transformation — a form of data leakage that would produce overly optimistic evaluation results.

### 6.3 Class Imbalance Handling

Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training set only:

```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
```

SMOTE generates synthetic minority-class samples by interpolating between existing fraud examples in feature space, creating a more balanced training distribution without simply duplicating minority examples.

**Before SMOTE (training set):** ~227,449 legitimate / ~394 fraud  
**After SMOTE (training set):** ~227,449 legitimate / ~227,449 fraud (balanced)

The test set is **never resampled**. It retains the original class distribution so that evaluation metrics reflect real-world conditions.

### 6.4 Data Leakage Prevention Summary

| Operation | Correct Approach |
|---|---|
| Train-test split | Before all transformations |
| StandardScaler | fit_transform on train; transform on test |
| SMOTE | Applied to train only; never to test |
| Model selection | CV on train; final evaluation once on test |

---

## 7. Machine Learning Models

### 7.1 Logistic Regression

Logistic Regression models the probability of a binary outcome using a sigmoid function applied to a linear combination of input features. It is computationally efficient, provides directly interpretable coefficients, and serves as a strong linear baseline. `class_weight='balanced'` is passed to compensate for class imbalance at the model level (in addition to SMOTE).

### 7.2 Decision Tree

A Decision Tree classifier recursively partitions the feature space by selecting splits that maximise information gain (or minimise Gini impurity). It is highly interpretable and requires no feature scaling. However, it is prone to overfitting on training data, which is expected to limit its generalisation performance. `class_weight='balanced'` is applied.

### 7.3 Random Forest

Random Forest is a bagging ensemble of decorrelated decision trees. Each tree is trained on a bootstrap sample of the training data, and predictions are aggregated by majority vote. The decorrelation is achieved by restricting each split to a random subset of features. Random Forest is typically robust to overfitting, handles class imbalance well with `class_weight='balanced'`, and produces reliable feature importance estimates via mean decrease in impurity.

### 7.4 Gradient Boosting

Gradient Boosting builds an ensemble sequentially: each tree is trained to predict the residual errors of the preceding ensemble. The scikit-learn `GradientBoostingClassifier` is used. It is a powerful algorithm on tabular data and often achieves strong performance on imbalanced classification problems. Because it does not natively support `class_weight`, the imbalance is handled through the upstream SMOTE step.

---

## 8. Model Evaluation

### 8.1 Why Accuracy is Insufficient

Given the class imbalance (~99.83 % legitimate), a naive classifier that always predicts "Legitimate" achieves ~99.83 % accuracy but detects zero fraud cases. Accuracy is therefore a misleading metric and is reported only for completeness.

### 8.2 Metrics Used

**Precision:**  
Of all transactions predicted as fraud, the proportion that are actually fraudulent.  
`Precision = TP / (TP + FP)`  
High precision means fewer false alarms — important for operational efficiency.

**Recall (Sensitivity):**  
Of all actual fraudulent transactions, the proportion that the model correctly identifies.  
`Recall = TP / (TP + FN)`  
High recall means fewer missed frauds — critically important for financial loss prevention.

**F1-score:**  
Harmonic mean of Precision and Recall.  
`F1 = 2 × (Precision × Recall) / (Precision + Recall)`  
Balances both concerns into a single scalar.

**ROC-AUC:**  
Area under the Receiver Operating Characteristic curve (True Positive Rate vs. False Positive Rate at all thresholds). Measures the classifier's ability to rank fraudulent transactions above legitimate ones. A value of 1.0 indicates perfect discrimination; 0.5 indicates random guessing.

**PR-AUC (Average Precision):**  
Area under the Precision-Recall curve. Unlike ROC-AUC, PR-AUC is not influenced by the large number of true negatives and is therefore more informative for highly imbalanced datasets. It directly measures the trade-off between catching more frauds (recall) and maintaining quality of fraud alerts (precision).

### 8.3 Confusion Matrix

For the selected model, the confusion matrix identifies:

| | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | True Negative (TN) | False Positive (FP) |
| **Actual Fraud** | False Negative (FN) | True Positive (TP) |

- **False Negative:** A fraudulent transaction incorrectly classified as legitimate. The fraud goes undetected. This is typically the most costly error in financial fraud — direct financial loss to the cardholder or issuing bank.
- **False Positive:** A legitimate transaction incorrectly flagged as fraud. The customer may experience a declined transaction or card block. This erodes customer trust and incurs operational costs.

The relative cost of FN vs FP is context-dependent. Most fraud detection systems accept a higher FP rate to minimise FN.

---

## 9. Experimental Results

> The following table is populated by executing the notebook. Values represent actual model predictions on the held-out test set (20 % of the dataset, stratified).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) |
| Decision Tree | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) |
| Random Forest | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) |
| Gradient Boosting | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) |
| **Best Model (Tuned)** | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) | (from notebook) |

> Run the notebook end-to-end to replace the placeholders with real computed values.

---

## 10. Final Model Selection

The final model is selected based on the highest combination of Recall, F1-score, and PR-AUC on the test set after initial comparison, with the understanding that:

1. **Recall is the primary objective.** Missing a fraud is more costly than a false alarm.
2. **Precision must be kept reasonable.** A model with 100 % Recall that flags every transaction as fraud is operationally useless.
3. **PR-AUC provides the most holistic view** given the class imbalance.

The selected model is tuned using `RandomizedSearchCV` with stratified 5-fold cross-validation on the training set. The test set is touched exactly once — for final evaluation after tuning is complete.

**Selection criterion (explicit statement):**

> *"The best model was selected based on the balance between Recall, Precision, F1-score, and PR-AUC rather than accuracy alone, prioritising fraud detection capability over raw classification accuracy."*

---

## 11. Threshold Analysis

The standard classification threshold of 0.5 is not necessarily optimal for fraud detection. By lowering the threshold (e.g., to 0.3 or 0.2), the classifier predicts fraud more aggressively — increasing Recall at the cost of Precision. By raising the threshold, the opposite effect is observed.

The notebook plots Precision, Recall, and F1-score as a function of threshold from 0.1 to 0.9. The optimal threshold is selected as the point that maximises F1-score (or alternatively maximises Recall subject to a minimum Precision constraint specified by the business).

This analysis demonstrates that threshold selection is a business decision, not a purely technical one. A financial institution with strict loss-minimisation goals would select a lower threshold; a system prioritising customer experience might accept a higher threshold.

---

## 12. Feature Importance and Interpretability

For tree-based models (Random Forest, Gradient Boosting), feature importance is derived from the mean decrease in impurity across all trees. The top 15 features are visualised in a horizontal bar chart.

For Logistic Regression, feature importance is approximated by the absolute value of the standardised coefficients.

**Interpretation caveat:** Features `V1`–`V28` are anonymised PCA components. While the model may assign high importance to `V14`, `V4`, or `V12` (commonly observed in published analyses), their meaning in terms of original transaction attributes (merchant type, time of day, geographic location, etc.) cannot be recovered from the published dataset.

`Amount` is the only feature with direct monetary interpretation. `Time` reflects the elapsed time since the dataset's first transaction, and its cyclic nature may capture day/night transaction patterns.

---

## 13. Limitations

1. **Anonymised features:** The inability to interpret `V1`–`V28` limits actionable insight for domain experts.
2. **Static, historical dataset:** The model is trained on 2013 data. Fraud patterns evolve (concept drift), and the model's performance would degrade without periodic retraining on fresh labelled data.
3. **Geographic and temporal scope:** Transactions are from European cardholders over two days. Generalisation to other geographies or time periods is not guaranteed.
4. **No transaction context:** Features such as merchant category, cardholder spending history, device fingerprint, and geolocation are absent. Modern fraud detection systems incorporate these signals.
5. **Threshold dependency:** Model performance is sensitive to the classification threshold. An operationally deployed system would require continuous threshold recalibration.
6. **SMOTE limitations:** SMOTE generates synthetic samples via interpolation. In high-dimensional, PCA-transformed space, these synthetic points may not accurately represent real fraud patterns.
7. **Not production-ready:** The saved pipeline is suitable for offline inference demonstrations only. A production system would require monitoring, versioning, serving infrastructure, and compliance checks.

---

## 14. Applications

- **Banking:** Automated screening of card-not-present (CNP) transactions.
- **Payment gateways:** Real-time transaction scoring in payment processing networks.
- **Digital wallets:** Fraud detection in mobile payment platforms (Apple Pay, Google Pay).
- **E-commerce:** Protecting merchants and customers from fraudulent online purchases.
- **Financial institutions:** Monitoring internal transfer anomalies and account takeover attempts.
- **Insurance:** Detecting fraudulent claims with similar binary classification framing.

---

## 15. Future Scope

1. **Real-time fraud detection:** Deploy the model as a low-latency scoring service (sub-100ms) integrated with transaction processing pipelines using Apache Kafka or AWS Kinesis.
2. **Online/incremental learning:** Algorithms such as Hoeffding Trees or River-based models that update continuously without full retraining.
3. **Concept drift detection:** Monitor prediction confidence distributions over time using drift detectors (ADWIN, Page-Hinkley) to trigger retraining.
4. **Deep learning:** Autoencoders for unsupervised anomaly detection; sequence models (LSTM, Transformer) to model cardholder spending trajectories.
5. **Graph-based fraud detection:** Model transactions as nodes and shared attributes (email, device, IP) as edges; apply Graph Neural Networks to detect fraud rings.
6. **Explainable AI (XAI):** Integrate SHAP or LIME for per-prediction feature attribution to support investigator workflows.
7. **Cost-sensitive learning:** Explicitly encode the asymmetric cost of FN (missed fraud loss) vs FP (customer friction cost) into the loss function.
8. **Federated learning:** Train models across multiple bank data silos without sharing raw transaction data.

---

## 16. Conclusion

This project demonstrates a complete, methodologically rigorous supervised learning pipeline for credit card fraud detection. Four classification algorithms were trained and evaluated under identical conditions on an 80/20 stratified split of the MLG-ULB dataset. SMOTE was applied exclusively to training data to address the severe class imbalance without contaminating the test set. Feature scaling was applied post-split to prevent data leakage. Model selection was based on Recall, F1-score, and PR-AUC rather than accuracy, reflecting the real-world cost asymmetry of fraud detection.

The project illustrates key ML engineering principles: preventing data leakage, selecting appropriate evaluation metrics for imbalanced classification, understanding the precision-recall trade-off, and performing threshold analysis to align model behaviour with operational requirements. The final trained pipeline is persisted and demonstrated to produce predictions on new transaction inputs.

While the resulting model is not production-ready — it lacks real-time infrastructure, monitoring, and drift detection — it constitutes a technically sound and academically defensible foundation for further development.

---

## References

[1] A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification," in *Proc. IEEE Symp. Series on Computational Intelligence*, Cape Town, 2015.

[2] A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 29, no. 8, pp. 3784–3797, Aug. 2018.

[3] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[5] T. Fawcett, "An Introduction to ROC Analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006.

[6] J. Davis and M. Goadrich, "The Relationship Between Precision-Recall and ROC Curves," in *Proc. 23rd Int. Conf. Machine Learning*, Pittsburgh, PA, 2006, pp. 233–240.

[7] G. Lemaître, F. Nogueira, and C. K. Aridas, "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning," *Journal of Machine Learning Research*, vol. 18, no. 17, pp. 1–5, 2017.

[8] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[9] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[10] Machine Learning Group — ULB, "Credit Card Fraud Detection Dataset," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
