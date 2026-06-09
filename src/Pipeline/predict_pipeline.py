import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            proba = model.predict_proba(data_scaled)[:, 1]

            return preds, proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 SeniorCitizen: int,
                 Partner: str,
                 Dependents: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 TotalCharges_log: float,
                 tenure_group: str,
                 charge_group: str,
                 is_month_to_month: int,
                 is_electronic_check: int,
                 is_alone: int,
                 high_risk_combo: int):

        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.TotalCharges_log = TotalCharges_log
        self.tenure_group = tenure_group
        self.charge_group = charge_group
        self.is_month_to_month = is_month_to_month
        self.is_electronic_check = is_electronic_check
        self.is_alone = is_alone
        self.high_risk_combo = high_risk_combo

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "TotalCharges_log": [self.TotalCharges_log],
                "tenure_group": [self.tenure_group],
                "charge_group": [self.charge_group],
                "is_month_to_month": [self.is_month_to_month],
                "is_electronic_check": [self.is_electronic_check],
                "is_alone": [self.is_alone],
                "high_risk_combo": [self.high_risk_combo],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
