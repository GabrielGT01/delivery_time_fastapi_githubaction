

"""
This class handles:
- Reading train/test CSV
- Validating column consistency
- Detecting data drift using KS-test
"""

import os
import sys
import json

import pandas as pd
import numpy as np
from pymongo import MongoClient  
from dataclasses import dataclass  
from src.logger import logging
from typing import List

from src.constants.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
)
from src.constants import training_pipeline_names
from src.constants.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from scipy.stats import ks_2samp  # For data drift detection
from src.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(training_pipeline_names.SCHEMA_FILE_PATH)
            logging.info("Schema file loaded successfully.")
        except Exception as e:
            logging.error(f"Error in initializing DataValidation: {e}")
            raise e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Loaded data from {file_path} with shape {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {e}")
            raise e

    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Parse the schema correctly - it's a list of single-key dictionaries
            required_columns = []
            for column_dict in self._schema_config["columns"]:
                required_columns.extend(column_dict.keys())
            
            logging.info(f"Required columns: {required_columns}")
            logging.info(f"DataFrame columns: {list(dataframe.columns)}")

            if set(required_columns) == set(dataframe.columns):
                logging.info("Column validation passed.")
                return True
            logging.warning("Column validation failed.")
            return False
        except Exception as e:
            logging.error(f"Error validating columns: {e}")
            raise e

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                if d1.dtype != d2.dtype:
                    logging.warning(f"Skipping drift check for {column} due to mismatched dtypes.")
                    continue

                ks_result = ks_2samp(d1, d2)
                drift_detected = ks_result.pvalue < threshold

                if drift_detected:
                    status = False

                report[column] = {
                    "p_value": float(ks_result.pvalue),
                    "drift_status": drift_detected,
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            logging.info(f"Drift report saved at {drift_report_file_path}")

            return status

        except Exception as e:
            logging.error(f"Error detecting dataset drift: {e}")
            raise e

    def start_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            if not self.validate_columns(train_df):
                raise Exception("Train dataset failed column validation.")

            if not self.validate_columns(test_df):
                raise Exception("Test dataset failed column validation.")

            drift_status = self.detect_dataset_drift(train_df, test_df)

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            logging.info("Validated train/test files saved successfully.")

            return DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            logging.error(f"Error in start_data_validation: {e}")
            raise e

