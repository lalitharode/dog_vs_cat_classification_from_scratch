import os
import sys
import cv2
import pathlib
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

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
            # Calculate the absolute path of the dog and cat images
            data_file_path_dog = os.path.join(os.getcwd(), 'notebooks', 'data', 'dog')
            data_file_path_cat = os.path.join(os.getcwd(), 'notebooks', 'data', 'cat')
            
            data_dir_dog = pathlib.Path(data_file_path_dog)
            data_dir_cat = pathlib.Path(data_file_path_cat)
            
            dog_cat_dict = {
                'dog': list(data_dir_dog.glob('*'))[0:500],  # Adjusted to capture all images in the dog folder
                'cat': list(data_dir_cat.glob('*'))[0:500]   # Adjusted to capture all images in the cat folder
            }

            dog_cat_label_dict = {'dog': 1, 'cat': 0}
            # Read the dataset
            x, y = [], []
            
            for pat_name, images in dog_cat_dict.items():
                for image in images:
                    img = cv2.imread(str(image))
                    if img is None:
                        logging.warning(f"Failed to load image: {image}")
                        continue
                    resize_image = cv2.resize(img, (180, 180))
                    x.append(resize_image)
                    y.append(dog_cat_label_dict[pat_name])
            
            # Check if any images were loaded
            if len(x) == 0:
                logging.error('No images loaded. Exiting the ingestion process.')
                return None, None

            x=np.array(x)
            y=np.array(y)
            
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
           
            X_train_scaled=X_train/255
            X_test_scaled=X_test/255
            
            n_samples_train, height, width, channels = X_train.shape
            n_samples_test = X_test.shape[0]
           
           
            flattened_X_train = X_train_scaled.reshape(n_samples_train, height * width * channels)
            flattened_X_test = X_test_scaled.reshape(n_samples_test, height * width * channels)

            # Save raw data to artifacts
            logging.info('Train test split completed.')

            # Split the data into train and test sets
            df_X_train = pd.DataFrame({
                  'image_array': [arr.tolist() for arr in flattened_X_train],  # Convert each row to a list
                'label': y_train  # Add labels as a new column
                })
            
            df_X_test = pd.DataFrame({
              'image_array': [arr.tolist() for arr in flattened_X_test],  # Convert each row to a list
                'label': y_test  # Add labels as a new column
                })
            
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save train and test data to artifacts
            df_X_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df_X_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f'Train data saved at: {self.ingestion_config.train_data_path}')
            logging.info(f'Test data saved at: {self.ingestion_config.test_data_path}')
            print(f"Train data saved at: {self.ingestion_config.train_data_path}")
            print(f"Test data saved at: {self.ingestion_config.test_data_path}")

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
