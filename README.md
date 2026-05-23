# Customer Churn Prediction using Machine Learning

## Project Overview
Customer churn is one of the biggest challenges for telecom companies.  
This project aims to predict whether a customer is likely to leave the company using Machine Learning techniques.

The project includes:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Handling Imbalanced Data
- Model Building
- Hyperparameter Tuning
- Model Evaluation

---

# Problem Statement
The objective is to identify customers who are likely to churn so that businesses can take preventive actions and improve customer retention.

---

# Dataset
Dataset Used: **Telco Customer Churn Dataset**

The dataset contains customer information such as:
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn Status

Target Variable:
- `Churn`
  - Yes → Customer left
  - No → Customer stayed

---

# Exploratory Data Analysis (EDA)

Performed detailed EDA to understand:
- Customer demographics
- Churn distribution
- Correlation between features and churn
- Numerical feature distributions
- Categorical feature analysis

Visualizations used:
- Histograms
- Countplots
- Boxplots
- Heatmaps

Key Insights:
- Customers with month-to-month contracts had higher churn
- Customers with shorter tenure were more likely to churn
- Higher monthly charges increased churn probability
- Customers without partners/dependents showed higher churn

---

# Feature Engineering

Feature engineering techniques applied:
- Label Encoding
- One-Hot Encoding
- Feature Transformation
- Handling Missing Values
- Scaling Numerical Features

Additional engineered features:
- Customer tenure groups
- Service categories
- Charge segmentation

---

# Handling Imbalanced Data

The dataset was imbalanced, so special attention was given to evaluation metrics.

Techniques used:
- `class_weight='balanced'`
- `scale_pos_weight` in XGBoost

Evaluation metrics focused on:
- Recall
- F1-score
- Precision

instead of relying only on accuracy.

---

# Machine Learning Models Used

The following models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier

---

# Hyperparameter Tuning

Hyperparameter tuning was performed using:
- GridSearchCV
- Cross Validation

Parameters tuned:
- max_depth
- learning_rate
- n_estimators
- min_samples_split
- regularization parameters
- class weights

---

# Best Model

## XGBoost Classifier

XGBoost achieved the best performance for the imbalanced dataset with:
- Highest Recall
- Best F1-score
- Balanced performance

This makes it the most suitable model for identifying churn customers.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

