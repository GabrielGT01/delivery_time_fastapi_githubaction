
import os
import sys
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.constants.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from src.constants.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def initiate_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("start data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.start_data_ingestion()
            logging.info(f"data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            logging.error(f"Error in ingesting data : {e}")
            print(f"\n❌ ingestion error: {e}")
            raise e

    def initiate_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            logging.info("Start validating data")
            data_validation_artifact = data_validation.start_data_validation()
            logging.info("Data validation completed")
            return data_validation_artifact
        except Exception as e:
            logging.error(f"Error in validating data : {e}")
            print(f"\n❌ validation error: {e}")
            raise e

    def initiate_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            logging.info('start transforming data')
            data_transformation_artifact = data_transformation.start_data_transformation()
            logging.info("Transformation completed")
            return data_transformation_artifact
        except Exception as e:
            logging.error(f'error in transforming data:{e}')
            print(f"\n❌ transformation error: {e}")
            raise e

    def initiate_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.start_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            logging.error(f'error in model trainer: {e}')
            print(f"\n❌ model trainer error: {e}")
            raise e

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.initiate_data_ingestion()
            data_validation_artifact = self.initiate_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.initiate_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.initiate_model_trainer(data_transformation_artifact=data_transformation_artifact)
        except Exception as e:
            raise e

        
