# Time-Series Ensemble Classification (Applied Machine Learning)

An applied machine learning project that builds a **general-purpose time-series
classification pipeline** using feature engineering, time-aware validation,
and ensemble learning.

This repository focuses on **methodology, evaluation, and modeling design**
rather than domain-specific assumptions or production deployment.

---

## 30-Second Quick View

This project demonstrates how to approach **binary classification problems on
time-ordered data** using an ensemble-based strategy.

The pipeline combines:
- Generic time-series feature engineering
- Multiple base learners with different inductive biases
- TimeSeries-aware cross-validation
- Probability-level blending (stacking)

### Core Skills Demonstrated:
Time-Series Analysis · Feature Engineering · Ensemble Learning · Model Evaluation · Backtesting · Applied Machine Learning


---

## Problem Framing

Given a univariate time series, the task is to predict a **binary outcome**
based on historical observations, such as:

- Whether the signal will increase or decrease over a future horizon
- Whether an event will occur in the next time step
- Directional or state-change classification problems

The framework is intentionally **domain-agnostic** and can be applied to
operational metrics, sensor readings, behavioral signals, or other sequential data.

---

## Feature Engineering

The feature set is constructed using **generic time-series transformations**,
including:

- Lagged values and first-order differences
- Percentage changes
- Rolling window statistics (mean, std, min, max)
- Rolling z-scores
- Short-vs-long rolling mean gaps as trend proxies

All features are derived from a single numeric series and do not rely on
domain-specific assumptions.

---

## Modeling Approach

### Base Models

Multiple complementary classifiers are trained as base learners:

- Logistic Regression (with normalization)
- Random Forest
- Gradient Boosting Decision Trees
- XGBoost (optional, if available)

Each model captures different structural patterns in the data.

---

### Ensemble Strategy (Blending)

To improve robustness, the project uses **probability-level blending**:

1. TimeSeriesSplit is used to generate out-of-fold predictions on the training set
2. Out-of-fold probabilities from base models are used as meta-features
3. A logistic regression model is trained as a second-level blender
4. The blended model is evaluated on a strictly held-out test set

This approach reduces overfitting risk and leverages model diversity.

---

## Evaluation Strategy

Evaluation follows **time-aware validation principles**:

- Chronological train / test split
- TimeSeriesSplit for cross-validation
- Metrics focused on classification quality rather than point accuracy

Primary evaluation metrics include:
- **Balanced Accuracy**
- **ROC-AUC**

Decision thresholds can be adjusted to support cost-sensitive analysis.

---

## Code Structure

```text
time-series-ensemble-classification/
├── time_series_modeling.py
└── README.md

```
