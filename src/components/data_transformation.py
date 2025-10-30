import os
import sys

import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
   def __init__(self):
      self.data_transformation_config = DataTransformationConfig()

   def get_data_preprocessor_obj(self):
      logging.info("creating data preprocessor object")
      try:
         
         pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
         ])

         logging.info("data preprocessor object created successfully")

         return pipeline

      except Exception as e:
         raise CustomException(e, sys)

   def initiate_data_transformation(self, train_path, test_path):
      try:
         train_data = pd.read_csv(train_path)
         test_data = pd.read_csv(test_path)
         logging.info("read train and test data completed")

         preprocessor_obj = self.get_data_preprocessor_obj()
         logging.info("obtained preprocessor object")

         target_column_name = "Target"

         input_feature_train_data = train_data.drop(columns=[target_column_name], axis=1)
         target_feature_train_data = train_data[target_column_name]

         input_feature_test_data = test_data.drop(columns=[target_column_name], axis=1)
         target_feature_test_data = test_data[target_column_name]

         input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
         input_test_arr = preprocessor_obj.transform(input_feature_test_data)
         logging.info("transformation on train and test data completed")
         
         train_arr = np.c_[input_train_arr, np.array(target_feature_train_data)]
         test_arr = np.c_[input_test_arr, np.array(target_feature_test_data)]
         logging.info("combining input features and target feature arrays completed")

         save_object(
            file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
         )
         logging.info("preprocessor object saved successfully")
         return(
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
         )

      except Exception as e:
         raise CustomException(e, sys)

