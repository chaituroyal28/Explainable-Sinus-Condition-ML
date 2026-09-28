# Explainable Machine Learning for Sinus Condition Prediction

## Project Title

Explainable Machine Learning for Sinus Condition Prediction Using Demographic, Symptom, Allergy and Environmental Factors

## Overview

This project applies machine learning and explainable artificial intelligence techniques to a survey-based sinus condition dataset.

The objective is to investigate whether demographic, symptom, allergy, environmental and healthcare-related survey variables can be used to predict self-reported sinus condition status.

The target variable is:

`Diagnosed with Sinus Condition`

The target represents self-reported survey information and is not a medically validated clinical diagnosis.

## Dataset

- Original records: 1,006
- Records after duplicate removal: 1,000
- Variables: 11
- Predictive features: 10
- Target variable: 1

## Features

1. Age
2. Gender
3. Regular Sinus Issues
4. Most Problematic Season
5. Triggered by External Factors
6. Known Allergy
7. Doctor Consulted (Past Year)
8. Sinus Severity (1-5)
9. Headaches or Facial Pain
10. Medication Frequency

## Machine Learning Models

The project evaluates:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost
- CatBoost

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Five-fold stratified cross-validation and randomized hyperparameter tuning are also used.

## Explainable AI

Two explainability approaches are included:

- Permutation Importance
- SHAP

The tuned Decision Tree is interpreted using SHAP.

## Results

The final held-out test results obtained during the project are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 51.00% | 50.93% | 55.00% | 52.88% | 49.63% |
| Decision Tree | 48.50% | 48.92% | 68.00% | 56.90% | 53.68% |
| Random Forest | 46.00% | 46.36% | 51.00% | 48.57% | 46.12% |
| SVM | 51.50% | 51.58% | 49.00% | 50.26% | 50.94% |
| XGBoost | 46.50% | 46.79% | 51.00% | 48.80% | 47.97% |
| CatBoost | 44.50% | 44.76% | 47.00% | 45.85% | 47.04% |

The Decision Tree achieved the highest F1-score on the held-out test set at 56.90%.

## SHAP

The largest mean absolute SHAP value was obtained for Age:

`0.051637`

Other relatively influential encoded features included:

- Regular Sinus Issues — Yes: 0.026117
- Headaches or Facial Pain — Yes: 0.019355
- Sinus Severity (1-5): 0.018509
- Headaches or Facial Pain — No: 0.013388
- Triggered by External Factors — No: 0.012760
- Known Allergy — Yes: 0.011923

SHAP values describe model behavior and should not be interpreted as causal effects.

## Project Structure

```text
Sinus_ML_Project/
│
├── data/
│   └── sinus_survey_data1.csv
│
├── notebooks/
│   └── sinus_ml_project.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── README.md
├── requirements.txt
└── .gitignore