import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def feature_engineering(self, df):
        try:
            logging.info("Starting feature engineering")

            # Fix TotalCharges — coerce errors to NaN then fill 0
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            )
            # FIXED LINE BELOW: Removed inplace=True
            df["TotalCharges"] = df["TotalCharges"].fillna(0)

            # Drop useless columns
            df.drop(
                columns=["customerID", "gender",
                         "PhoneService", "MultipleLines",
                         "StreamingTV", "StreamingMovies"],
                axis=1, inplace=True, errors='ignore'
            )

            # Log transform TotalCharges
            df["TotalCharges_log"] = np.log1p(df["TotalCharges"])
            df.drop(columns=["TotalCharges"], inplace=True)

            # tenure_group feature
            def get_tenure_group(t):
                if t <= 10:   return "Danger"
                elif t <= 30: return "Transition"
                elif t <= 60: return "Established"
                else:         return "Loyal"

            df["tenure_group"] = df["tenure"].apply(get_tenure_group)
            df.drop(columns=["tenure"], inplace=True)

            # charge_group feature
            def get_charge_group(c):
                if c < 30:   return "Basic"
                elif c < 60: return "Mid"
                elif c < 85: return "High_Risk"
                else:        return "Premium"

            df["charge_group"] = df["MonthlyCharges"].apply(get_charge_group)
            df.drop(columns=["MonthlyCharges"], inplace=True)

            # Clean internet service columns
            service_cols = [
                "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport"
            ]
            for col in service_cols:
                df[col] = df[col].replace("No internet service", "No")

            # Binary risk features
            df["is_month_to_month"] = (
                df["Contract"] == "Month-to-month"
            ).astype(int)

            df["is_electronic_check"] = (
                df["PaymentMethod"] == "Electronic check"
            ).astype(int)

            df["is_alone"] = (
                (df["Partner"] == "No") &
                (df["Dependents"] == "No")
            ).astype(int)

            df["high_risk_combo"] = (
                (df["tenure_group"] == "Danger") &
                (df["charge_group"].isin(["High_Risk", "Premium"]))
            ).astype(int)

            # Encode target
            if "Churn" in df.columns:
                df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

            logging.info(f"Feature engineering done. Shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, X):
        """Build preprocessor pipeline based on actual columns"""
        try:
            categorical_columns = X.select_dtypes(
                include="object"
            ).columns.tolist()

            numerical_columns = X.select_dtypes(
                exclude="object"
            ).columns.tolist()

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )),
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test data")

            # Feature engineering on both
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            target_column = "Churn"

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            # Build preprocessor on train columns
            preprocessor = self.get_data_transformer_object(X_train)

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info("Preprocessing complete. Saving object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)