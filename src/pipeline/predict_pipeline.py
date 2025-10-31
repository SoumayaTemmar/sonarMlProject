import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
   def __init__(self):
      pass

   def predict(self, features):
      try:
         # model path and preprocessor path
         model_path = "artifacts/model.pkl"
         preprocessor_path = "artifacts/preprocessor.pkl"

         # load both model and preprocessor
         model = load_object(model_path)
         preprocessor = load_object(preprocessor_path)

         #features preprocessing
         preprocessed_features = preprocessor.transform(features)

         #predict the output
         Y_pred = model.predict(preprocessed_features)

         return Y_pred
      except Exception as e:
         raise CustomException(e, sys)
      
class CustomData:
   def __init__(self, *features):

      self.features = features

   def get_data_as_data_frame(self):
      try:
         data = {
            f"Feature_{i}": [self.features[i]] for i in range(len(self.features))
         }
         data_frame = pd.DataFrame(data)
         return data_frame

      except Exception as e:
         raise CustomException(e, sys)

