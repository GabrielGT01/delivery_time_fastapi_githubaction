
import os
import sys
import json
import certifi
import pymongo


import pandas as pd
import numpy as np
from pymongo import MongoClient  # Changed to direct import for better Python 3.12 compatibility
from dataclasses import dataclass  # Missing import for dataclass
from src.logger import logging
from typing import List
from sklearn.model_selection import train_test_split

from src.constants.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.constants.artifact_entity import DataIngestionArtifact


from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class DataIngestion():
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            
            
        except Exception as e:
            raise Exception(f"{e}") 

    def read_transform_dataframe(self):

        """
        read data from mongo_db
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            logging.info('database and collection succesfuly found')
            #the datas are in json format, list then make it a dataframe
            df = pd.DataFrame(list(collection.find()))
            print(f"Loaded {len(df)} documents from MongoDB")


            if "_id" in df.columns.to_list():
                df=df.drop(columns = ["_id"], axis = 1)
                
            #convert all nan to numpy nan
            df.replace({"na":np.nan}, inplace = True)

            return df
            
        except Exception as e:
            logging.error(f"Error in reading and transforming to dataframe : {e}")
            raise e
    

    def save_data_to_machine(self,dataframe:pd.DataFrame):

        """
        collect the data, read and save the data to local host machine
        """
        
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            ##creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok = True)
            logging.info("succesfully created the folder")

            
            dataframe.to_csv(feature_store_file_path, index= False, header = True)
            logging.info("dataframe succesfully converted and stored in local host")

            
            return dataframe

            
            
        except Exception as e:
            logging.error(f"Error in storing data: {e}")
            raise e
            
    def split_data(self, dataframe:pd.DataFrame):

        """
        colect the data, then slit into test and train data

        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("dataframe successfully splitted")

            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok = True)
            logging.info("created trainfolder")

            
            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            logging.info("saved training file in the created folder")
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index = False, header = True )
            logging.info("saved test file in the test folder")
            
            
            
        except Exception as e:
            logging.error(f"Error in storing data: {e}")
            raise e
            
        

    def start_data_ingestion(self):
        try:
            dataframe = self.read_transform_dataframe()
            
            dataframe = self.save_data_to_machine(dataframe)
            self.split_data(dataframe)

            dataingestionartifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path
            )
            
            return dataingestionartifact
        except Exception as e:
            logging.error(f"Error in storing data: {e}")
            raise e
            



if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)

        
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("initiating reading and downloading data")
        dataingestionartifact=data_ingestion.start_data_ingestion()

        print("Train file:", dataingestionartifact.train_file_path)
        print("Test file:", dataingestionartifact.test_file_path)

    except Exception as e:
        logging.error(f"error in ingesting file {e}")
        raise e
