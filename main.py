import os
import sys
import json
import certifi
import pymongo
import pandas as pd
import numpy as np
from pymongo import MongoClient  
from dataclasses import dataclass  
from src.logger import logging
from typing import List
from sklearn.model_selection import train_test_split
from src.constants.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.constants.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from scipy.stats import ks_2samp  # For data drift detection
from src.utils import read_yaml_file, write_yaml_file
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":  # Fixed: Changed ** to __
    try:
        logging.info("Starting ML Training Pipeline")
        
        # Data Ingestion
        logging.info("Initializing Training Pipeline Configuration")
        trainingpipelineconfig = TrainingPipelineConfig()
        
        logging.info("Initializing Data Ingestion Configuration")
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        
        logging.info("Initiating reading and downloading data")
        dataingestionartifact = data_ingestion.start_data_ingestion()
        print("Train file:", dataingestionartifact.train_file_path)
        print("Test file:", dataingestionartifact.test_file_path)
        logging.info("Data ingestion fully downloaded from mongo db database")
        
        # Data Validation
        print("Start validation of data")
        logging.info("Initializing Data Validation Configuration")
        data_valid_config = DataValidationConfig(trainingpipelineconfig)
        
        # datavalid inherits from both sides
        data_validation = DataValidation(dataingestionartifact, data_valid_config)
        logging.info("Start validating data")
        data_validation_artifact = data_validation.start_data_validation()
        logging.info("Data validation completed")
        print("Data Validation Result:", data_validation_artifact)
        
        # Check if data validation passed
        if not data_validation_artifact.validation_status:
            logging.error("Data validation failed. Pipeline stopped.")
            raise Exception("Data validation failed")
        
        # Data Transformation
        logging.info("Starting data transformation")
        data_transform_config = DataTransformationConfig(trainingpipelineconfig)
        data_transform = DataTransformation(data_validation_artifact, data_transform_config)
        data_transformation_artifact = data_transform.start_data_transformation()
        print("Data Transformation Result:", data_transformation_artifact)
        logging.info("Data transformation completed")
        
        # Model Training
        logging.info("Starting model training")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.start_model_trainer()
        print("Model Training Result:", model_trainer_artifact)
        logging.info("Model training completed")
        
        # Final Summary
        logging.info("="*60)
        logging.info("ML TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        logging.info(f"Final Model Path: {model_trainer_artifact.trained_model_file_path}")
        logging.info(f"Training R2 Score: {model_trainer_artifact.train_metric_artifact.r2_score}")
        logging.info(f"Test R2 Score: {model_trainer_artifact.test_metric_artifact.r2_score}")
        logging.info(f"Training MAE: {model_trainer_artifact.train_metric_artifact.mean_absolute_error}")
        logging.info(f"Test MAE: {model_trainer_artifact.test_metric_artifact.mean_absolute_error}")
        logging.info("="*60)
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"✓ Data Ingestion: Completed")
        print(f"✓ Data Validation: {'Passed' if data_validation_artifact.validation_status else 'Failed'}")
        print(f"✓ Data Transformation: Completed")
        print(f"✓ Model Training: Completed")
        print(f"✓ Final Model R2 Score: {model_trainer_artifact.test_metric_artifact.r2_score:.4f}")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Error in ML training pipeline: {e}")
        print(f"\n❌ Pipeline failed with error: {e}")
        raise e
