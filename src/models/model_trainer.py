import os
import sys
from dataclasses import dataclass
from pathlib import Path
import mlflow
from mlflow.sklearn import log_model
import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from src.utils.exception import CustomException
from src.utils.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    output_folder: Path = Path.cwd() / "models"
    trained_model_file_path: Path = output_folder / "model.pkl"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            mlflow.start_run(run_name='mlops_practice')  # Start MLflow run
            mlflow.log_param("output_folder", str(self.model_trainer_config.output_folder))
            mlflow.log_param("trained_model_file_path", str(self.model_trainer_config.trained_model_file_path))

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }
            params = {
                "Random Forest": {
                    'max_features': 5,
                    'n_estimators': 50
                },
                "Gradient Boosting": {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                },
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name].fit(X_train, y_train)

            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_param("best_model_score", best_model_score)

            # Log hyperparameters
            for param_name, param_value in params[best_model_name].items():
                mlflow.log_param(f"{best_model_name}_{param_name}", param_value)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_scr = r2_score(y_test, predicted)
            mlflow.log_metric("MAE", mean_absolute_error(y_test, predicted))
            mlflow.log_metric("R2_score", r2_scr)
            print(f'MAE is {mean_absolute_error(y_test, predicted)}')
            print(f'R2 score is {r2_scr}')

            # Log the model using MLflow
            log_model(best_model, "best_model")

            return r2_scr

        except Exception as e:
            raise CustomException(e, sys)
        finally:
            mlflow.end_run()  # End MLflow run
