# Telco Customer Churn Prediction

**Live Demo:** [telco-customer-churn-predictior.onrender.com](https://telco-customer-churn-predictior.onrender.com)  

---

## Project Overview

End-to-end machine learning pipeline that predicts whether a telecom customer will churn, built with a **production-grade modular architecture** — not just a notebook. The system ingests raw customer data, applies feature engineering, trains and evaluates multiple models, and serves predictions through a Flask web app deployed via Docker on Render.

**Business problem:** Acquiring a new customer costs 5–10× more than retaining an existing one. This system flags at-risk customers *before* they leave, giving the retention team time to act.

---

## Results

| Model | Recall (Churn) | F1 Score | Accuracy |
|-------|---------------|----------|----------|
| **XGBoost** ✅ | **0.76** | **0.79** | **81%** |
| Random Forest | 0.68 | 0.72 | 79% |
| Logistic Regression | 0.71 | 0.70 | 78% |
| Decision Tree | 0.63 | 0.65 | 75% |

**Why Recall over Accuracy?**  
The dataset has a 1:3 class imbalance (26.5% churn). A model predicting "No Churn" for everyone scores 73% accuracy — completely useless. Missing a churner costs the business far more than a false alarm, so **Recall on the churn class** is the primary metric.

---

## Dataset

**Source:** [Telco Customer Churn — IBM Sample Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- **7,043 customers**, 21 raw features
- **Churn rate:** 26.5% (Yes) vs 73.5% (No)
- Features include: tenure, contract type, payment method, internet services, monthly/total charges

---

## Key EDA Insights

```
Insight 1 — New customers churn most
  Customers in their first 10 months have 50%+ churn rate.
  After month 60, churn drops to near zero.
  → Created tenure_group feature: Danger / Transition / Established / Loyal

Insight 2 — Month-to-month contracts are the biggest churn signal
  Month-to-month customers churn at 42% vs 11% on annual contracts.
  → Created is_month_to_month binary feature

Insight 3 — Electronic check = high churn
  Electronic check users churn at 45% vs ~15% for auto-pay methods.
  → Created is_electronic_check binary feature

Insight 4 — TotalCharges was right-skewed (skewness: 9.4)
  Log transform (log1p) applied → skewness reduced to 0.3
  Prevents high-value outliers from dominating distance calculations

Insight 5 — Several features had near-zero predictive value
  Gender, PhoneService, MultipleLines, StreamingTV, StreamingMovies
  dropped after churn rate analysis showed <2% difference across categories
```

---

## Feature Engineering (8 engineered features)

| Feature | Logic | Why |
|---------|-------|-----|
| `tenure_group` | Danger / Transition / Established / Loyal | Non-linear churn vs tenure relationship |
| `charge_group` | Basic / Mid / High_Risk / Premium | Charge-churn correlation captured in buckets |
| `TotalCharges_log` | log1p(TotalCharges) | Removes right skew (9.4 → 0.3) |
| `is_month_to_month` | Contract == 'Month-to-month' | Single biggest churn predictor |
| `is_electronic_check` | PaymentMethod == 'Electronic check' | 45% churn rate in this group |
| `is_alone` | No Partner AND No Dependents | Isolated customers churn more |
| `high_risk_combo` | Danger tenure + High_Risk/Premium charges | Compounded risk signal |
| Dropped raw columns | tenure, MonthlyCharges, TotalCharges | Replaced by engineered features |

---

## Handling Class Imbalance (1:3 ratio)

```python
# XGBoost — scale positive weight
XGBClassifier(scale_pos_weight=3)  # ratio of negative:positive

# sklearn models — class weight
RandomForestClassifier(class_weight='balanced')
LogisticRegression(class_weight='balanced')
```

No SMOTE/oversampling used — the 1:3 ratio is manageable without synthetic data. Threshold tuned at 0.55 (not default 0.5) to further improve Recall.

---

## Project Architecture

```
telco-customer-churn-prediction/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py         # loads CSV → artifacts/train.csv + test.csv
│   │   ├── data_transformation.py    # feature engineering + sklearn Pipeline
│   │   └── model_trainer.py          # GridSearchCV across 4 models
│   ├── pipeline/
│   │   └── predict_pipeline.py       # loads pkl artifacts → inference
│   ├── logger.py                     # timestamped log files per run
│   ├── exception.py                  # custom traceback with file + line number
│   └── utils.py                      # save/load pickle helpers + evaluate_models
├── notebook/
│   ├── tele.ipynb                    # EDA + feature engineering exploration
│   └── data/teleco.csv               # raw dataset
├── templates/
│   └── index.html                    # Flask frontend
├── artifacts/                        # auto-generated — model.pkl, preprocessor.pkl
├── logs/                             # auto-generated — timestamped .log files
├── app.py                            # Flask application (entry point)
├── main.py                           # training pipeline entry point
├── Dockerfile                        # containerization
├── requirements.txt
└── setup.py                          # find_packages() setup
```

---

## sklearn Pipeline

```python
# Numerical features
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical features
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combined preprocessor
preprocessor = ColumnTransformer([
    ("num_pipeline", num_pipeline, numerical_columns),
    ("cat_pipeline", cat_pipeline, categorical_columns)
])
```

The **preprocessor is saved as a .pkl artifact** — ensuring identical transformations at training and inference. No data leakage.

---

## Model Training

**GridSearchCV** was used to tune all 4 models simultaneously. XGBoost won on F1 score.

```python
models = {
    "XGBoost": XGBClassifier(scale_pos_weight=3, ...),
    "Random Forest": RandomForestClassifier(class_weight='balanced', ...),
    "Logistic Regression": LogisticRegression(class_weight='balanced', ...),
    "Decision Tree": DecisionTreeClassifier(...)
}
# Best model auto-selected by F1 score on test set
# Saved to artifacts/model.pkl
```

---

## Setup & Run Locally

**1. Clone and install**
```bash
git clone https://github.com/yashprateek1712/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
pip install -r requirements.txt
```

**2. Add dataset**

Place `teleco.csv` in `notebook/data/`

**3. Train the models**
```bash
python main.py
```
This runs the full pipeline and saves `artifacts/model.pkl` and `artifacts/preprocessor.pkl`

**4. Start the web app**
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000)

**5. Run with Docker**
```bash
docker build -t telco-churn .
docker run -p 5000:5000 telco-churn
```

---

## Test Cases

**Likely to churn** (high risk):

| tenure | Contract | MonthlyCharges | PaymentMethod |
|--------|----------|----------------|---------------|
| 3 months | Month-to-month | ₹85 | Electronic check |

**Likely to stay** (low risk):

| tenure | Contract | MonthlyCharges | PaymentMethod |
|--------|----------|----------------|---------------|
| 48 months | Two year | ₹45 | Bank transfer |

---

## Tech Stack

`Python` · `XGBoost` · `Scikit-learn` · `Pandas` · `NumPy` · `Seaborn` · `Flask` · `Docker` · `Render`

---

## Author

**Prateek (Yash) Srivastav**  
B.Tech Electrical Engineering + Minor in AI · NIT Patna  
[LinkedIn](https://linkedin.com/in/prateek-srivastav1712) · [GitHub](https://github.com/yashprateek1712) · [LeetCode](https://leetcode.com/u/yashprateek1712)
