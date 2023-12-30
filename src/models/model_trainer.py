import os
import sys
from dataclasses import dataclass
import numpy as np

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils.exception import CustomException
from src.utils.utils import save_object, evaluate_models
from pathlib import Path

@dataclass
class ModelTrainerConfig:
    output_folder: Path = Path.cwd() / "models"
    trained_model_file_path: Path = output_folder / "model.pkl"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "extra tree" : ExtraTreesRegressor()
            }
            params = {
                "Decision Tree": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    'max_features':5
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    'max_features': 5,
                    'n_estimators': 50
                },
                "extra tree": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    'max_features': 5
                }

            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # best_model = models[best_model_name]
            best_model = RandomForestRegressor().fit(X_train,y_train)
            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_scr = r2_score(y_test, predicted)
            print(f'MAE is {mean_absolute_error(y_test, predicted)}')
            print(f'R2 score is {r2_scr}')
            return r2_scr


        except Exception as e:
            raise CustomException(e, sys)