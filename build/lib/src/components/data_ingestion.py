import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# from src.components.data_transformation import DataTransformation

## Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join(os.getcwd(), 'artifacts', 'train.csv')
    test_data_path: str = os.path.join(os.getcwd(), 'artifacts', 'test.csv')
    raw_data_path: str = os.path.join(os.getcwd(), 'artifacts', 'raw.csv')

## Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Start')
        try:
            # Calculate the absolute path of gemstone.csv based on your current directory
            data_file_path = os.path.join(os.getcwd(), 'notebooks', 'data', 'gemstone.csv')

            # Read the dataset
            df = pd.read_csv(data_file_path)
            logging.info('Dataset read as pandas DataFrame')

            # Save raw data to artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Train test split')

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save train and test data to artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    # print(f"Train data saved at: {train_data_path}")
    # print(f"Test data saved at: {test_data_path}")
    
    # below steps are for data_transformation pkl file which we can get directly from data_transformation file also 
    # # and from here as well
    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    
    