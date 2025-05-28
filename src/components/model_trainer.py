
import os
import sys
import mlflow
from mlflow.models.signature import infer_signature

from src.constants.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig
)

from src.constants.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.logger import logging
from src.utils import save_object, save_numpy_array_data, load_object, load_numpy_array_data, evaluate_models, get_regression_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import DeliveryModel


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Starting ModelTrainer initialization")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
            # Configure MLflow tracking URI to use local file store
            mlflow.set_tracking_uri("file:./mlruns")
            logging.info("MLflow tracking URI set to local file store")
            
            logging.info("ModelTrainer initialization completed successfully")
        except Exception as e:
            logging.error(f"Error in ModelTrainer initialization: {e}")
            raise e

    def track_mlflow(self, best_model, metric, signature):
        try:
            with mlflow.start_run():
                r2_score = metric.r2_score
                mean_absolute_error_score = metric.mean_absolute_error
                mean_squared_error_score = metric.mean_squared_error
                
                mlflow.log_metric('r2_score', r2_score)
                mlflow.log_metric('mean_absolute_error_score', mean_absolute_error_score)
                mlflow.log_metric('mean_squared_error_score', mean_squared_error_score)
                
                # Log model with signature
                mlflow.sklearn.log_model(
                    sk_model=best_model, 
                    artifact_path="model",
                    signature=signature
                )
                
                logging.info("MLflow tracking completed successfully")
        except Exception as e:
            logging.error(f"Error in MLflow tracking: {e}")
            # Don't raise exception here to avoid breaking the pipeline
            logging.warning("Continuing without MLflow tracking...")

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training process")
            logging.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            logging.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            
            models = {
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            }
            logging.info(f"Models initialized: {list(models.keys())}")
            
            params = {
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'positive': [False, True]
                },
                
                "XGBRegressor": {
                    'n_estimators': [100, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                },
                
                "CatBoosting Regressor": {
                    'iterations': [200, 500],
                    'depth': [6, 8],
                    'learning_rate': [0.05, 0.1],
                }
            }
            logging.info("Hyperparameter grids defined for all models")
            
            logging.info("Starting model evaluation with GridSearchCV")
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                               models=models, param=params)
            logging.info(f"Model evaluation completed. Results: {model_report}")

            # Get best score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            logging.info(f"Best model identified: {best_model_name} with score: {best_model_score}")
            
            # Make predictions with the best model (already fitted by evaluate_models)
            logging.info("Making predictions with the best model")
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            logging.info("Calculating regression metrics")
            regression_train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
            regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
            
            logging.info(f"Training metrics: {regression_train_metric}")
            logging.info(f"Test metrics: {regression_test_metric}")

            # MLflow tracking - wrap in try-except to prevent pipeline failure
            try:
                # Track train results with MLflow
                train_signature = infer_signature(X_train, y_train_pred)
                self.track_mlflow(best_model, regression_train_metric, train_signature)

                # Track test results with MLflow
                test_signature = infer_signature(X_test, y_test_pred)
                self.track_mlflow(best_model, regression_test_metric, test_signature)
            except Exception as mlflow_error:
                logging.warning(f"MLflow tracking failed: {mlflow_error}")
                logging.info("Continuing model training without MLflow tracking...")

            # Load preprocessor
            logging.info(f"Loading preprocessor from: {self.data_transformation_artifact.transformed_object_file_path}")
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessor loaded successfully")
                
            # Create model directory
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            logging.info(f"Model directory created/verified: {model_dir_path}")

            # Create DeliveryModel object
            logging.info("Creating DeliveryModel object")
            delivery_model = DeliveryModel(preprocessor=preprocessor, model=best_model)
            
            # Save the trained model
            logging.info(f"Saving trained model to: {self.model_trainer_config.trained_model_file_path}")
            save_object(self.model_trainer_config.trained_model_file_path, obj=delivery_model)
            logging.info("Trained model saved successfully")

            # Model pusher
            final_model_dir = "final_model"
            os.makedirs(final_model_dir, exist_ok=True)
            save_object("final_model/model.pkl", best_model)
            logging.info("Final model saved to final_model/model.pkl")

            # Create Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            logging.info("Model training process completed successfully")
            
            print("Training Metrics:")
            print(model_trainer_artifact.train_metric_artifact)
            print('---' * 20)
            print("Test Metrics:")
            print(model_trainer_artifact.test_metric_artifact)
            
            return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"Error in train_model method: {e}")
            raise e
        
    def start_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer pipeline")
            
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path 
            
            logging.info(f"Loading training data from: {train_file_path}")
            logging.info(f"Loading test data from: {test_file_path}")
            
            # Loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            logging.info(f"Training array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            # Split features and target
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            
            logging.info("Data split completed - features and target separated")
            logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

            # Train model with all required parameters
            logging.info("Calling train_model method")
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            
            logging.info("Model trainer pipeline completed successfully")
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in start_model_trainer method: {e}")
            raise e
