# 🚗 Uber Ride Cancellation Prediction
**Author:** Junghoon Park

**Project:** End-to-end data analytics and machine learning project to predict ride cancellations using the Uber NCR Ride Bookings dataset.

---

## 📌 Project Overview

Ride cancellations are a common issue for ride-hailing platforms. This project builds a **pre-ride cancellation prediction system** that identifies high-risk rides at booking time using only information available before the ride begins.

The key constraint: the model avoids using any post-ride features preventing data leakage and keeping the system deployable in practice.

---

## 📂 Dataset

**Source:** [Uber Ride Analytics Dashboard — Kaggle](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard/data)

**File:** `ncr_ride_bookings.csv`  
**Size:** 150,000 bookings across 148,788 unique customers

| Column | Description |
|--------|-------------|
| Date | Date of the booking |
| Time | Time of the booking |
| Booking ID | Unique identifer for each booking |
| Booking_Status | Status of booking (Completed, Cancelled by Driver/Customer, etc.) |
| Customer ID | Unique identifier for customers |
| Vehicle_Type | Type of vehicle |
| Pickup_Location | Starting location of the ride |
| Drop_Location | Destination location of the ride |
| Avg_VTAT | Average vehicle time to arrive at pickup (mins) |
| Avg_CTAT | Average trip duration from pickup to destination (mins) |
| Cancelled Rides by Customer | Customer cancellation flag |
| Reason for cancelling by Customer | Customer cancellation reason |
| Cancelled Rides by Driver | Driver cancellation flag |
| Driver Cancellation Reason | Driver cancellation reason |
| Incomplete Rides | Incomplete ride flag |
| Incomplete Rides Reason | Reason for Incomplete rides |
| Ride Distance | Distance covered during the ride (km) |
| Booking_Value | Fare amount  |
| Driver_Rating | Driver rating |
| Customer_Rating | Customer rating |
| Payment Method | Payment Method |

---

## 🗂️ Notebook Structure

```
1. Data Overview          → shape, dtypes, basic info
2. Missing Value Analysis → null patterns and visualization
3. Data Preparation       → duplicate analysis, feature engineering, renaming
4. Univariate Analysis    → booking status, vehicle type distributions
5. Cancellation Analysis  → reasons, data availability, feature impact
6. Key Findings           → EDA summary, ML motivation
7. Data Preprocessing     → feature selection, encoding, train/test split, class weight
8. Model Training         → Logistic Regression, Random Forest, XGBoost
9. Model Evaluation       → comparison table, feature importance
10. Conclusion            → findings, limitations, and potential improvements
```

## 📊 Exploratory Data Analysis (EDA)

Key observations from the data:

- Completed rides account for ~62% of bookings, indicating moderate class imbalance  
- Driver cancellations occur ~2.6× more frequently than customer cancellations  
- Avg_VTAT shows a clear relationship with cancellations (longer wait → higher cancellation rate)  
- No single dominant cancellation reason, cancellations are distributed across multiple factors  
These insights guided feature selection and model design.
---
## ⚙️ Approach

### Target Variable
```
Is_Cancelled = 1  → Cancelled by Driver / Cancelled by Customer / No Driver Found / Incomplete
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

### Model Preparation
- Train/test split first (80/20, stratified) before any target encoding and imputation
- Target encoding computed on training data only and applied to test
- Addressed class imbalance (62/38) addressed by computing class weights using 'compute_class_weight'

---

## 📊 Results

| Model | AUC | Recall | Precision | F1 |
|-------|-----|--------|-----------|-----|
| Logistic Regression | 0.506 | 0.501 | 0.383 | 0.434 |
| Random Forest | 0.718 | 0.626 | 0.514 | 0.565 |
| **XGBoost** | **0.720** | **0.640** | **0.514** | **0.570** |

**Best Model: XGBoost**
- Highest AUC (0.720) and Recall (0.640) across all metrics
- Recall improved significantly with imbalance handling
- Consistent performance with Random Forest confirms result reliability

---

## 🔍 Key Findings

**1. Driver cancellations dominate**
Driver cancellations (18%) occur 2.6x more frequently than customer cancellations (7%) : driver behavior is the primary cancellation signal.

**2. Avg_VTAT is the strongest predictor**
Vehicle arrival time dominates feature importance across all three models. Longer wait times directly increase cancellation likelihood.

**3. Temporal and vehicle features are weak**
Hour, Weekday, and Vehicle_Type show relatively small variation in cancellation rate and are weak individual predictors confirmed by both EDA and feature importance.


---

## ⚠️ Model Limitations

The moderate performance is not a modelling failure rather it reflects the fundamental constraints of pre-ride prediction:

- Pre-ride features carry weak predictive signal
- Most informative features (Driver_Rating, Booking_Value) are provided after completing ride. Thus, including them would constitute data leakage. 
- Cancellations are situational with no single dominant pattern

**Performance could improve with more predictive features:**
- Customer cancellation history
- Driver reliability / acceptance rate
- Real-time traffic and weather conditions
- Time elapsed since driver assignment

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
