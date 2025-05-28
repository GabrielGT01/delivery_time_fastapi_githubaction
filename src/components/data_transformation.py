
import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from src.constants.training_pipeline_names import (
    TARGET_COLUMN, 
    numeric_features,
    categorical_features,
    target_features,
    traffic_order
)

from src.constants.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig
)

from src.constants.artifact_entity import (
    DataValidationArtifact, 
    DataTransformationArtifact
)
from src.logger import logging
from src.utils import save_object, save_numpy_array_data


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            
        except Exception as e:
            raise e
    
    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise e
    
    def get_data_transformer_object(self):
        """
        This function creates and returns the preprocessing object with imputation
        """
        logging.info("Accessing encoders and imputers")
        
        try:
            # Numeric pipeline: Impute with mean then scale
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            # Categorical pipeline: Impute with mode then encode
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
            
            # Target encoding pipeline: Impute with mode then target encode
            target_encoder_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('target_encoder', TargetEncoder())
            ])
            
            # Traffic ordinal encoding pipeline: Impute with mode then ordinal encode
            traffic_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=traffic_order, handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            
            logging.info("Initiating encoders and scaling")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numeric', numeric_transformer, numeric_features),
                    ('categorical', categorical_transformer, categorical_features),
                    ('target_encoded', target_encoder_transformer, target_features),
                    ('traffic', traffic_transformer, ['Traffic_Level'])
                ],
                remainder='passthrough'
            )
            
            return preprocessor
            
        except Exception as e:
            logging.error(f'Could not transform the inputs: {e}')
            raise Exception(f"Error in creating preprocessor: {e}")
    
    def start_data_transformation(self):
        """
        This method initiates the data transformation with imputation
        """
        logging.info("Starting data transformation")
        
        try:
            # Read training and testing data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info("Read both test and train data successfully")
            
            # Check for missing values before transformation
            logging.info(f"Train data missing values:\n{train_df.isnull().sum()}")
            logging.info(f"Test data missing values:\n{test_df.isnull().sum()}")
            
            # Separate features and target variable
            input_feature_train_df = train_df.drop(columns=["Order_ID", TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info("Split the train data into features and target")
            
            input_feature_test_df = test_df.drop(columns=["Order_ID", TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Split the test data into features and target")
            
            # Get the preprocessor object
            preprocessor = self.get_data_transformer_object()
            
            # Apply transformations
            logging.info("Applying transformations on training data")
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df, target_feature_train_df)
            
            logging.info("Applying transformations on test data")
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)
            
            # Combine transformed features with target variable
            train_arr = np.c_[
                transformed_input_train_feature, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                transformed_input_test_feature, np.array(target_feature_test_df)
            ]
            
            logging.info("Transformation completed successfully")
            logging.info(f"Transformed train array shape: {train_arr.shape}")
            logging.info(f"Transformed test array shape: {test_arr.shape}")
            
            # Save transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            
            save_object("final_model/preprocessor.pkl",preprocessor)
            
            # Prepare data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise Exception(f"Data transformation failed: {e}")
