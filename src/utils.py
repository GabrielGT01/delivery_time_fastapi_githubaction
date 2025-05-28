
import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
import joblib
import yaml
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.constants.artifact_entity import RegressionMetricArtifact
from src.constants.training_pipeline_names import SAVED_MODEL_DIR, MODEL_FILE_NAME
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        logging.info("Exited the save_object method class")
    except Exception as e:
        raise e

def load_object(file_path: str) -> object:
    try:
        logging.info(f"Loading object from: {file_path}")
        
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        
        with open(file_path, "rb") as file_obj:
            obj = joblib.load(file_obj)
            logging.info(f"Successfully loaded object from: {file_path}")
            return obj
            
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise Exception(f"Error loading object from {file_path}: {e}") from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise e


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for model_name in models.keys():  # More readable iteration
            model = models[model_name]
            para = param[model_name]
            
            # Perform GridSearchCV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            # Set the best parameters and fit the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store test score in report
            report[model_name] = test_model_score
            
        return report
        
    except Exception as e:
        raise Exception(f"Error in evaluating models: {e}")  # Fixed error message

def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        model_r2_score = r2_score(y_true, y_pred)
        model_mean_absolute_error_score = mean_absolute_error(y_true, y_pred)
        model_mean_squared_error_score = mean_squared_error(y_true, y_pred)
        regression_metric = RegressionMetricArtifact(
            r2_score=model_r2_score,
            mean_absolute_error=model_mean_absolute_error_score,
            mean_squared_error=model_mean_squared_error_score
        )
        return regression_metric
    except Exception as e:
        raise e

class DeliveryModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise Exception(f"Error initializing DeliveryModel: {e}")
        
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise Exception(f"Error in prediction: {e}")
