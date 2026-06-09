import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("Starting the Machine Learning Training Pipeline...")

        # Step 1: Data Ingestion
        logging.info(">>> Initiating Data Ingestion <<<")
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        logging.info(">>> Initiating Data Transformation <<<")
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_data_path, 
            test_data_path
        )

        # Step 3: Model Training
        logging.info(">>> Initiating Model Training <<<")
        trainer = ModelTrainer()
        best_model_score = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Pipeline Completed Successfully! Best Model Metric Score: {best_model_score}")
        print(f"Success! Model trained with score: {best_model_score}")
        print(f"Check the 'artifacts' folder for preprocessor.pkl and model.pkl")

    except Exception as e:
        logging.error("Pipeline failed.")
        raise CustomException(e, sys)