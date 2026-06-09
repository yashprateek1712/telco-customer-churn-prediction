import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score
)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Classification models — class_weight for imbalance
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    random_state=42
                ),
                "XGBoost": XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42
                ),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1],
                    "class_weight": ["balanced", None],
                },
                "Decision Tree": {
                    "max_depth": [4, 6, 8, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "class_weight": ["balanced"],
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 8, 10, 15],
                    "min_samples_leaf": [1, 2, 4],
                    "class_weight": ["balanced"],
                },
                "XGBoost": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "scale_pos_weight": [3],  # handles 1:3 imbalance
                    "subsample": [0.7, 0.8, 1.0],
                },
            }

            model_report: dict = evaluate_models(
                x_train=X_train, y_train=y_train,
                x_test=X_test,  y_test=y_test,
                models=models, param=params
            )

            logging.info(f"Model report: {model_report}")

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Threshold for churn — f1 > 0.55 on minority class
            if best_model_score < 0.55:
                raise CustomException("No good model found — f1 too low")

            logging.info(
                f"Best model: {best_model_name} "
                f"with F1 score: {best_model_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final evaluation
            predicted = best_model.predict(X_test)
            logging.info(
                "\n" + classification_report(y_test, predicted)
            )

            return f1_score(y_test, predicted)

        except Exception as e:
            raise CustomException(e, sys)