import os
import sys

import pandas as pd
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
   train_data_path = os.path.join("artifacts","train.csv")
   test_data_path = os.path.join("artifacts", "test.csv")
   raw_data_path = os.path.join("artifacts", "data.csv")


class DataIngestion:
   def __init__(self):
      self.data_ingestion_config = DataIngestionConfig()


   def initiate_data_ingestion(self):
      logging.info("entring data ingetion component")
      try:
         
         df = pd.read_csv("sonar_data_NoteBook\data\sonar data.csv", header=None)
         df.columns = [f"Feature_{i}" for i in range(60)] + ["Target"]
         logging.info("reading dataset completed")

         os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
         df.to_csv(self.data_ingestion_config.raw_data_path, index=False,header=True)

         
         train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
         logging.info("spliting data into trainset and testset completed")

         
         train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
         test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
         logging.info("data ingestion completed")

         return(
            self.data_ingestion_config.train_data_path,
            self.data_ingestion_config.test_data_path
         )

      except Exception as e:
         raise CustomException(e, sys)

if __name__ == "__main__":
   obj = DataIngestion()
   train_data_path, test_data_path = obj.initiate_data_ingestion()

   data_transformation = DataTransformation()
   train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

   model_trainer = ModelTrainer()
   acc_score, class_report,_ = model_trainer.initiate_model_trainer(train_arr, test_arr)

   print(f"accuracy: {acc_score},\n classification report:\n {class_report}")

