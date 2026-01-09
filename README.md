# Credit Risk Prediction Using Cost-Sensitive Machine Learning and Fairness Analysis

This project builds a complete **credit risk prediction pipeline** to classify loan applicants as *good* or *bad* credit risks using exploratory data analysis, cost-sensitive learning, interpretable modelling, and fairness diagnostics.
The objective is to minimise the **financial cost of high-risk misclassification** while maintaining equitable performance across demographic groups.

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)

A comprehensive EDA was conducted to understand dataset structure, class imbalance, financial behaviour patterns, and demographic distributions.

Key insights:

* The dataset contains **mixed numerical and categorical financial features** describing applicant behaviour.
* The *bad credit* class is the **minority**, requiring imbalance-aware evaluation.
* Features such as **credit history**, **duration**, **credit amount**, **savings**, and **employment status** show strong predictive power.

Visualisations used include class distribution plots, numerical feature histograms, outlier boxplots, and group-wise comparisons across age and personal-status categories.

---

## Feature Engineering

A structured preprocessing pipeline was implemented using `ColumnTransformer` to ensure safe and leakage-free transformations:

* **Numerical Processing:**
  Standardisation using `StandardScaler`.

* **Categorical Encoding:**
  One-Hot Encoding using `OneHotEncoder(handle_unknown="ignore")`.

This enables consistent feature alignment between training and real-world inference.

---

## Machine Learning Models

Multiple models were trained and compared under a cost-sensitive credit-risk setting.

### Final Thesis Models (Locked)

* Logistic Regression — interpretable baseline
* Random Forest — non-linear ensemble model
* Gradient Boosting — high-capacity learner

### Evaluation Design (Cost-Aware)

* Stratified train/test split
* Risk-focused metrics:

  * Recall (Bad credit class)
  * F1-score (Bad credit class)
  * ROC-AUC
  * **Total Financial Cost**
* Confusion-matrix visualisation
* Decision-threshold tuning for financial cost optimisation

---

## Cost-Sensitive Learning

A misclassification cost matrix was defined:

| Actual \ Predicted | Good | Bad |
| ------------------ | ---- | --- |
| Good               | 0    | 1   |
| Bad                | 5    | 0   |

Failing to identify a bad applicant is therefore **five times more costly** than rejecting a good one.

Models were trained using:

* `class_weight` for Logistic Regression & Random Forest
* `sample_weight` for Gradient Boosting

---

## Results & Evaluation

Final cost-aware model performance:

| Model                                 | Accuracy | Recall (Bad) | F1-Score (Bad) | Total Cost |
| ------------------------------------- | -------: | -----------: | -------------: | ---------: |
| Logistic Regression (Threshold Tuned) |   0.6900 |       0.8556 |         0.6235 |    **145** |
| Random Forest (Tuned)                 |   0.7567 |       0.3556 |         0.4672 |        305 |

---

### Interpretation

* Threshold-tuned Logistic Regression achieved the **lowest financial cost**, despite lower overall accuracy.
* Random Forest, although more accurate, failed to capture a large proportion of bad-risk applicants, resulting in **significantly higher cost**.
* This confirms that **accuracy alone is misleading** in financial decision systems.

---

## Fairness Analysis

Post-training fairness diagnostics were conducted across:

* **Age Groups:** Young / Middle / Senior
* **Personal Status / Gender Proxy**

For each group, recall of the bad-risk class was measured to ensure improvements in cost do not disproportionately disadvantage specific demographic groups.

---

## Technology Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## Repository Files

* `DS_Project_IS.ipynb` — full notebook (EDA → preprocessing → cost-sensitive modelling → fairness analysis → threshold optimisation)

---

## How to Run the Project

```bash
pip install -r requirements.txt
jupyter notebook
# Open: DS_Project_IS.ipynb
```
