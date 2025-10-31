import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
   def __init__(self):
      pass

   def initiate_training(self):
      try:
         logging.info("training pipeline started")
         #data ingestion
         logging.info("data ingestion started")
         data_ingestion = DataIngestion()
         train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
         logging.info(f"data ingestion completed train_data_path:{train_data_path} test_data_path: {test_data_path}")

         #data transformation
         logging.info("data transformation started")
         data_transformation = DataTransformation()
         train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
         logging.info(f"data transformation completed preprocessor_path: {preprocessor_path}")

         #model trainer
         logging.info("model training started")
         model_trainer = ModelTrainer()
         acc_score, class_report,model_path = model_trainer.initiate_model_trainer(train_arr, test_arr)
         logging.info(f"model training completed model_path: {model_path}")

         return acc_score, class_report, model_path
         
      except Exception as e:
         raise CustomException(e, sys)
      
if __name__ == "__main__":
   train_pipeline = TrainPipeline()
   acc, report, model_path = train_pipeline.initiate_training()
   print("Accuracy Score:", acc)
   print("Classification Report:\n", report)
   print("Trained model saved at:", model_path)

      