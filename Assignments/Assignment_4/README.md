# Assignment 4: Heart Disease Classification Analysis

**Student:** Mohammad Waliullah Shah | P37000087
**University:** University of Naples Federico II
**Course:** Statistical Data Analysis
**Professors:** Prof. Roberta Siciliano & Prof. Giulia Vannucci
**Academic Year:** 2025-2026

---

## Objective
Classify heart disease presence using multiple classification models.

## Dataset
- Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- 303 observations, 14 features
- Binary target: 0 = No disease, 1 = Disease

## Steps Completed
- EDA: Class distribution, Pie chart, Correlation heatmap, Pair plots, Bar charts
- Preprocessing: Missing value imputation, StandardScaler, class_weight balanced
- Models: Logistic Regression, LDA, QDA, Decision Tree (CART)
- Evaluation: Confusion Matrix, ROC Curve, AUC

## Results
| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| **Logistic Regression** | **86.9%** | **0.867** | **0.950** |
| LDA | 85.3% | 0.848 | 0.940 |
| QDA | 86.9% | 0.875 | 0.944 |
| Decision Tree (CART) | 82.0% | 0.814 | 0.820 |

**Best Model: Logistic Regression (AUC = 0.950)**

## Key Findings
- `thal` is the most important feature (Gini = 0.39)
- `cp`, `ca`, `oldpeak` also strong predictors
- Linear models outperform Decision Tree on this dataset

## Files
- `assignment4_heart_disease.py` - Python code with comments
- `assignment4_report.txt` - Academic report
- `eda_plots/` - 8 visualization plots
- `heart.csv` - Dataset
