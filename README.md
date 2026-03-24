# 🚗 Uber Ride Cancellation Prediction
**Author:** Junghoon Park
End-to-end data analytics and machine learning project to predict ride cancellations using the Uber NCR Ride Bookings dataset.

---

## 📌 Project Overview

Ride cancellations are a major operational challenge for ride-hailing platforms. This project builds a **pre-ride cancellation prediction system** — identifying high-risk rides at booking time using only information available before the ride begins.

The key constraint: **no post-ride features are used**, ensuring the model is deployable in a real production environment without data leakage.

---

## 📂 Dataset

**Source:** [Uber Ride Analytics Dashboard — Kaggle](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard/data)

**File:** `ncr_ride_bookings.csv`  
**Size:** 150,000 bookings across 148,788 unique customers

| Column | Description |
|--------|-------------|
| Booking_Status | Ride outcome (Completed, Cancelled by Driver/Customer, etc.) |
| Vehicle_Type | Type of vehicle booked |
| Pickup_Location | Pickup zone |
| Drop_Location | Drop zone |
| Avg_VTAT | Average vehicle time to arrive at pickup (mins) |
| Avg_CTAT | Average customer time to arrive at pickup (mins) |
| Hour | Hour of booking |
| Weekday | Day of the week |
| Booking_Value | Fare amount (post-ride only) |
| Driver_Rating | Driver rating (post-ride only) |
| Ride_Distance | Trip distance (post-ride only) |

---

## 🗂️ Notebook Structure

```
1. Data Overview          → shape, dtypes, basic info
2. Missing Value Analysis → null patterns, missingno visualization
3. Data Preparation       → duplicate analysis, feature engineering, renaming
4. Univariate Analysis    → booking status, vehicle type distributions
5. Cancellation Analysis  → reasons, data availability, feature impact
6. Key Findings           → EDA summary, ML motivation
7. Data Preprocessing     → feature selection, encoding, train/test split
8. Model Training         → Logistic Regression, Random Forest, XGBoost
9. Model Evaluation       → comparison table, ROC curves, feature importance
10. Conclusion            → findings, limitations, recommendations
```

---

## ⚙️ Methodology

### Target Variable
```
Is_Cancelled = 1  → Cancelled by Driver / Customer / No Driver Found / Incomplete
Is_Cancelled = 0  → Completed
```

### Feature Selection
Only pre-ride features used to prevent data leakage:

| Feature | Type | Encoding |
|---------|------|----------|
| Vehicle_Type | Categorical | Label Encoding |
| Avg_VTAT | Numeric | Median Imputation |
| Hour | Numeric | As-is |
| Weekday | Categorical | Manual mapping (Mon=0 … Sun=6) |
| Pickup_Location | Categorical (150 unique) | Target Encoding |
| Drop_Location | Categorical (150 unique) | Target Encoding |

**Excluded (data leakage):** `Booking_Value`, `Ride_Distance`, `Driver_Rating`, `Customer_Rating`, `Avg_CTAT`, `Payment_Method`, cancellation reason columns.

### Preprocessing
- Train/test split first (80/20, stratified) — before any encoding
- Target encoding computed on training data only, applied to test
- Class imbalance (62/38) addressed via `class_weight='balanced'`

---

## 📊 Results

| Model | AUC | Recall | Precision | F1 |
|-------|-----|--------|-----------|-----|
| Logistic Regression | 0.506 | 0.501 | 0.383 | 0.434 |
| Random Forest | 0.718 | 0.626 | 0.514 | 0.565 |
| **XGBoost** | **0.720** | **0.640** | **0.514** | **0.570** |

**Best Model: XGBoost**
- Catches **64% of cancellations** using only pre-ride information
- Consistent performance with Random Forest confirms result reliability

---

## 🔍 Key Findings

**1. Driver cancellations dominate**
Driver cancellations (18%) occur 2.6x more frequently than customer cancellations (7%) — driver behavior is the primary cancellation signal.

**2. Avg_VTAT is the strongest predictor**
Vehicle arrival time dominates feature importance across all three models. Longer wait times directly increase cancellation likelihood.

**3. Temporal and vehicle features are weak**
Hour, Weekday, and Vehicle_Type show less than 3% variation in cancellation rate — weak individual predictors confirmed by both EDA and feature importance.

**4. Cancellations are multi-factorial**
No single dominant reason found — reasons are evenly distributed (~22-25% each), making cancellations inherently situational and harder to predict.

---

## ⚠️ Model Limitations

The ~0.72 AUC ceiling is not a modeling failure — it reflects the fundamental constraints of pre-ride prediction:

- Pre-ride features carry weak predictive signal by nature
- Most informative features (Driver_Rating, Booking_Value) are post-ride only — including them would constitute data leakage
- Cancellations are situational with no single dominant pattern

**Performance could improve with:**
- Customer cancellation history
- Driver reliability / acceptance rate
- Real-time traffic and weather conditions
- Time elapsed since driver assignment

---

## 💡 Business Recommendation

- **Reduce Avg_VTAT** — shorter driver arrival time is the single most impactful lever to reduce cancellations
- **Location-based intervention** — flag high-risk pickup zones for proactive driver allocation
- **Model deployment** — XGBoost can score rides at booking time to trigger early intervention for high-risk rides

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualizations |
| missingno | Missing value analysis |
| scikit-learn | Preprocessing, LR, RF, evaluation |
| XGBoost | Gradient boosting model |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/uber-cancellation-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn missingno scikit-learn xgboost plotly

# 3. Download dataset from Kaggle
# Place ncr_ride_bookings.csv in the project folder

# 4. Open and run the notebook
jupyter notebook Uber_Cancellation.ipynb
```

---

## 📁 Repository Structure

```
uber-cancellation-prediction/
├── Uber_Cancellation.ipynb    # Main notebook
├── README.md                  # This file
└── ncr_ride_bookings.csv      # Dataset (download from Kaggle)
```
