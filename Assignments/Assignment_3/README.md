# Assignment 3: Car Price Prediction using Regression Analysis

**Student:** Mohammad Waliullah Shah | P37000087  
**University:** University of Naples Federico II  
**Course:** Statistical Data Analysis  
**Professors:** Prof. Roberta Siciliano & Prof. Giulia Vannucci  
**Academic Year:** 2025-2026  

---

## Objective
Predict car prices using regression models (Linear, Ridge, Lasso).

## Dataset
- Source: [Kaggle - Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
- 205 observations, 26 features

## Steps Completed
- EDA: Histograms, Box plots, Correlation Heatmap, Scatter plots
- Feature Engineering: Brand extraction, Encoding, Log transform
- Models: Linear Regression, Ridge Regression, Lasso Regression
- Evaluation: R², MAE, RMSE

## Results
| Model | R² Test | MAE | RMSE |
|-------|---------|-----|------|
| Linear Regression | 0.907 | 0.122 | 0.159 |
| **Ridge Regression** | **0.928** | **0.115** | **0.139** |
| Lasso Regression | 0.926 | 0.111 | 0.141 |

**Best Model: Ridge Regression (R² = 0.928)**

## Files
- `assignment3_car_price.py` - Python code with comments
- `assignment3_report.txt` - Academic report
- `eda_plots/` - 8 visualization plots
- `CarPrice_Assignment.csv` - Dataset
