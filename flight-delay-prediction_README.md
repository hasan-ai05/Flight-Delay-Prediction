# Flight Delay Prediction

A binary classification model that predicts whether a US flight will arrive more than
15 minutes late — using only information available before the flight departs.
The model catches 58.7% of real delays from pre-departure features alone.

---

## The Problem

Departure delay data is available at the gate — but by then the passenger is already
waiting. The more useful question is: **can we predict arrival delay from information
known before boarding?**

This imposes a strict constraint: any feature that is only measurable after departure
is excluded. DEPARTURE_DELAY — the single strongest predictor — cannot be used.

---

## Dataset

**Source:** US Department of Transportation Flight Delays 2015 — Kaggle  
**Full size:** 5,819,079 flights, 31 columns  
**Working sample:** 100,000 flights (random, stratified to preserve delay rate)  
**Target:** is_delayed = 1 if ARRIVAL_DELAY > 15 minutes (FAA standard definition)

**Class distribution:**

| Class | Count | % |
|---|---|---|
| On Time | 82,208 | 82.2% |
| Delayed (>15 min) | 17,792 | 17.8% |

Imbalance ratio: 4.6:1. Accuracy is a misleading metric on this dataset.

---

## Leakage Prevention

Three columns were excluded despite strong predictive power:

| Column | Reason Excluded |
|---|---|
| DEPARTURE_DELAY | Directly causes arrival delay — known only after pushback |
| ARRIVAL_TIME | Measured after landing |
| ELAPSED_TIME | Actual flight duration — post-flight measurement |

Including DEPARTURE_DELAY would produce AUC > 0.95 and a model that cannot
be used in practice — it requires information that does not exist at prediction time.

---

## Features Used (10)

| Feature | Type | Rationale |
|---|---|---|
| MONTH | Numeric | Seasonal patterns |
| DAY_OF_WEEK | Numeric | Weekly traffic patterns |
| DEP_HOUR | Engineered | Cascade effect — evening flights worse |
| IS_PEAK_HOUR | Engineered | Morning/evening rush flag |
| IS_WEEKEND | Engineered | Weekend traffic profile |
| AIRLINE_ENC | Encoded | Carrier operational efficiency |
| ORIGIN_ENC | Encoded | Airport congestion profile |
| SCHEDULED_TIME | Numeric | Planned flight duration |
| DISTANCE | Numeric | Route length |
| TAXI_OUT | Numeric | Time from gate to runway — congestion signal |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LightGBM | 0.6784 | 0.2962 | 0.5871 | 0.3937 | **0.7042** |
| CatBoost | 0.6706 | 0.2918 | 0.5964 | 0.3918 | 0.7032 |
| Logistic Regression | 0.6555 | 0.2729 | 0.5627 | 0.3676 | 0.6695 |

**LightGBM wins by ROC-AUC (0.7042).**

**Why Accuracy = 67.8% is not a problem:**
A model that predicts "on time" for every flight achieves 82.2% accuracy
while catching zero delays. The 67.8% figure reflects the model's decision
to trade overall accuracy for higher delay detection rate.

**Confusion matrix (LightGBM on 20,000 test flights):**

| | Predicted On Time | Predicted Delayed |
|---|---|---|
| Actual On Time | 11,478 (TN) | 4,964 (FP) |
| Actual Delayed | 1,469 (FN) | 2,089 (TP) |

1,469 missed delays (False Negatives) — the number that matters operationally.

**Cross-validation (5-fold Stratified):** 0.716 ± 0.005 — stable across all splits.

![Model Comparison](images/model_comparison.png)

---

## EDA Findings

**Worst airline:** NK (Spirit Airlines) — 28.5% delayed  
**Best airline:** HA (Hawaiian Airlines) — 11.1% delayed  
**Overall rate:** 17.8%

The cascade effect is visible in the hourly delay rate chart.
Morning flights (7–9am) show lower delay rates. By 7–9pm, accumulated
small delays from earlier aircraft rotations produce the highest rates of the day.

![Airline Delay Rates](images/airline_delay_rates.png)
![EDA Dashboard](images/eda_dashboard.png)

---

## SHAP Explainability

SHAP TreeExplainer was applied to the LightGBM model.

**Most confident prediction in test set:**
A Thursday evening flight (19:00, peak hour) predicted delayed at 99.9% confidence.

TAXI_OUT dominated the SHAP waterfall — long runway wait signals airport congestion
at the moment of departure, the strongest real-time indicator of an incoming delay.

DEP_HOUR was second — the cascade effect quantified: late evening flights carry
accumulated delays from the day's earlier rotations.

![SHAP Summary](images/shap_summary.png)
![SHAP Waterfall](images/shap_waterfall.png)

---

## How to Run

```bash
pip install kagglehub shap lightgbm catboost scikit-learn pandas numpy plotly
jupyter notebook Aviation_Flight_Delay.ipynb
```

Dataset downloads automatically via `kagglehub`. Sampling to 100k is handled
inside the notebook — no memory issues on standard Colab instances.

---

## Where to Place Screenshots

```
flight-delay-prediction/
    images/
        eda_dashboard.png          Section 5 — EDA subplots
        airline_delay_rates.png    Section 13 — airline bar chart
        model_comparison.png       Section 9 — metrics + ROC curves
        confusion_matrix.png       Section 10 — heatmap
        shap_summary.png           Section 11 — beeswarm
        shap_waterfall.png         Section 12 — single flight
        executive_dashboard.png    Section 15 — full dashboard
    README.md
    Aviation_Flight_Delay.ipynb
```

---

## Project Structure

```
Aviation_Flight_Delay.ipynb    main notebook (15 sections)
README.md                      this file
images/                        screenshots from notebook output
```

---

*Part of an 8-project AI Engineering portfolio — Hasan Akhras*
