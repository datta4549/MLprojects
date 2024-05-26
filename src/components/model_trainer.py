import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay,precision_score, recall_score, f1_score, roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting train and test feature")
            X_train, y_train, X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = {
                    "Random Forest":RandomForestClassifier(),
                    "Decision Tree":DecisionTreeClassifier(),
                    "Gradient Boosting":GradientBoostingClassifier(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "Logistic Regression":LogisticRegression(),
                    "K-Neighbors Classifier":KNeighborsClassifier(),
                    "Support Vector Classifier":SVC(),
                    "XGBClassifier":XGBClassifier(),
                    "CatBoosting Classifier":CatBoostClassifier()
            }

            params = {
                "K-Neighbors Classifier": {
                    "n_neighbors":[2,3,10,20,40,50],
                    'metric':['minkowski','manhattan']
                    },
                "Decision Tree":{},
                "Gradient Boosting":{},
                "AdaBoost Classifier":{},
                "Logistic Regression":{},
                "K-Neighbors Classifier":{},
                "Support Vector Classifier":{},
                "Random Forest": {
                    'max_depth': [5, 15, 4,6,8],
                    'max_features': ['log2', 'sqrt'],
                    'criterion': ['gini', 'entropy'],
                    'n_estimators':[100,200,400,1000]},
                "XGBClassifier": {
                    "learning_rate": [0.1,0.01],
                    'n_estimators':[100,200,300],
                    "colsample_bytree":[0.5,0.8,1,0.3,0.4]},
                "CatBoosting Classifier": {
                    "learning_rate": [0.1,0.01],
                    'max_depth': [5, 15, 4,6,8]}
            }

            model_report:dict= evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test = y_test,models=models,param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            #best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            f1 = f1_score(y_test,predicted, average='weighted')
            
            return f1
        except Exception as e:
            raise CustomException(e,sys)