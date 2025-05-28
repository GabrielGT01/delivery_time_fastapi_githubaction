
## this file pushes the data from my local machine to the mongo db
import os
import sys
import json
import certifi
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from pymongo import MongoClient  # Changed to direct import for better Python 3.12 compatibility
from dataclasses import dataclass  # Missing import for dataclass
from src.logger import logging

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

# This creates a simple class that holds detail:
@dataclass
class DataExtractConfig:
    source_data_path: str = "Food_Delivery_Times.csv"
    database = "captgt007" #name of database on mongo db
    collection = "delivery_time" #this is a random name and can be changed
    
class DataExtract():
    def __init__(self):
        # inherit all from dataextractconfig
        self.ingestion_config = DataExtractConfig()
        
    def csv_to_json(self):
        try:
            # reads data and drops the index so it doesnt affect the json type 
            data = pd.read_csv(self.ingestion_config.source_data_path)
            data.reset_index(drop=True, inplace=True)
            # convert the csv file to json style format
            records = list(json.loads(data.T.to_json()).values())
            return records
        
        except Exception as e:
            logging.error(f"Error during data reading: {e}")
            raise Exception(f"Error loading the file path: {e}")
    
    def push_data_mongodb(self, records):
        try:
            database_name = self.ingestion_config.database
            collection_name = self.ingestion_config.collection
            
            # Use certifi for secure connection
            self.mongo_client = MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            
            database = self.mongo_client[database_name]
            collection = database[collection_name]  
            collection.insert_many(records)
            
            return len(records)
        
        except Exception as e:
            logging.error(f"Error shipping or uploading data: {e}")
            raise Exception(f"Error sending the data: {e}") 

if __name__ == "__main__":
    data_extract_object = DataExtract()
    records = data_extract_object.csv_to_json()
    print("Successfully transformed object from csv to json")
    no_of_records = data_extract_object.push_data_mongodb(records)
    print(f"Successfully inserted {no_of_records} records into MongoDB")
