
import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
   try:
      dir_path = os.path.dirname(file_path)
      os.makedirs(dir_path, exist_ok=True)

      with open(file_path, 'wb') as file_obj:
         dill.dump(obj, file_obj)
         
      logging.info("object saved successfully")
   except Exception as e:
      raise CustomException(e, sys)

def evaluate_models(x_train,y_train,x_test,y_test,models):
   results = []
   try:
      for name, model in models.items():
         cv_score = cross_val_score(model,x_train,y_train, cv=5)

         #fit the model and predict
         model.fit(x_train, y_train)
         y_pred = model.predict(x_test)
         acc_score = accuracy_score(y_test, y_pred)

         #append the results
         results.append({
            "model_name": name,
            "cv_mean_score":cv_score.mean(),
            "acc_score": acc_score
         })

      return results

   except Exception as e:
      raise CustomException(e,sys)
   

def load_object(file_path):

   try:
      with open(file_path, 'rb') as file_obj:
         return dill.load(file_obj)
   except Exception as e:
      raise CustomException(e, sys)
