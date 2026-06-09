from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import os
import sys

from src.Pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')

    try:
        # Derived features calculated from raw inputs
        tenure = int(request.form.get('tenure'))
        monthly_charges = float(request.form.get('MonthlyCharges'))
        total_charges = float(request.form.get('TotalCharges'))
        contract = request.form.get('Contract')
        payment = request.form.get('PaymentMethod')
        partner = request.form.get('Partner')
        dependents = request.form.get('Dependents')

        # Derive engineered features
        import numpy as np
        total_charges_log = np.log1p(total_charges)

        def get_tenure_group(t):
            if t <= 10:   return "Danger"
            elif t <= 30: return "Transition"
            elif t <= 60: return "Established"
            else:         return "Loyal"

        def get_charge_group(c):
            if c < 30:   return "Basic"
            elif c < 60: return "Mid"
            elif c < 85: return "High_Risk"
            else:        return "Premium"

        tenure_group = get_tenure_group(tenure)
        charge_group = get_charge_group(monthly_charges)
        is_month_to_month = 1 if contract == "Month-to-month" else 0
        is_electronic_check = 1 if payment == "Electronic check" else 0
        is_alone = 1 if (partner == "No" and dependents == "No") else 0
        high_risk_combo = 1 if (
            tenure_group == "Danger" and
            charge_group in ["High_Risk", "Premium"]
        ) else 0

        data = CustomData(
            SeniorCitizen=int(request.form.get('SeniorCitizen')),
            Partner=partner,
            Dependents=dependents,
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            Contract=contract,
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=payment,
            TotalCharges_log=total_charges_log,
            tenure_group=tenure_group,
            charge_group=charge_group,
            is_month_to_month=is_month_to_month,
            is_electronic_check=is_electronic_check,
            is_alone=is_alone,
            high_risk_combo=high_risk_combo
        )

        pred_df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        preds, proba = pipeline.predict(pred_df)

        churn = bool(preds[0])
        probability = round(float(proba[0]) * 100, 1)

        # Risk level
        if probability >= 70:
            risk = "High Risk"
            risk_class = "high"
        elif probability >= 40:
            risk = "Medium Risk"
            risk_class = "medium"
        else:
            risk = "Low Risk"
            risk_class = "low"

        result = {
            "churn": churn,
            "probability": probability,
            "risk": risk,
            "risk_class": risk_class,
            "tenure_group": tenure_group,
            "charge_group": charge_group,
        }

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
