import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
   trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
   def __init__(self):
      self.model_trainer_config = ModelTrainerConfig()

   def initiate_model_trainer(self, train_arr, test_arr):
      logging.info("entred the model trainer component")

      try:
         
         x_train, y_train, x_test, y_test = (
            train_arr[:,:-1],
            train_arr[:,-1],
            test_arr[:, :-1],
            test_arr[:,-1]
         )
         logging.info("train test split completed")

         models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(max_iter=1000)
         }

         # evaluate each and every model and get the scores
         results = evaluate_models(x_train, y_train, x_test, y_test,models)

         # get best model name
         
         best_obj = max(results, key = lambda x: x['acc_score'])
         best_model_name = best_obj['model_name']
         best_model_score = best_obj['acc_score']
         print(f"best_model_name: {best_model_name}")

         #get the model
         best_model = models[best_model_name]

         if best_model_score < 0.6:
            raise CustomException("no best model found !")
         
         save_object(
            file_path= self.model_trainer_config.trained_model_file_path,
            obj= best_model
         )

         y_pred = best_model.predict(x_test)
         acc_score = accuracy_score(y_test, y_pred)
         classification_rep = classification_report(y_test, y_pred)
         logging.info("model trainer component completed")
         
         return(
            acc_score,
            classification_rep,
            self.model_trainer_config.trained_model_file_path
         )
         

      except Exception as e:
         raise CustomException(e, sys)

