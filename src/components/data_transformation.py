import os
import cv2
import sys
import ast
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join(os.getcwd(), 'artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        # self.data_transformation_config = DataTransformationConfig()
        pass

    # @staticmethod
    # def image_preprocessor(images):
    #     """
    #     Preprocesses the images by scaling pixel values to [0, 1].
    #     """
    #     return np.array(images).astype('float32') / 255.0

    # def get_data_transformation_object(self):
    #     """
    #     Returns the image preprocessing object that handles normalization of image data.
    #     """
    #     try:
    #         logging.info('Image Data Transformation initiated')
    #         logging.info('Image preprocessing object created')
    #         return self.image_preprocessor

    #     except Exception as e:
    #         logging.error('Error in Image Data Transformation')
    #         raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process, including loading, normalizing, and saving processed data.
        """
        try:
            # Reading train and test CSV files
            df_X_train = pd.read_csv(train_path)
            df_X_test = pd.read_csv(test_path)

            logging.info('Loaded train and test datasets')

            image_height, image_width, num_channels = 180, 180, 3

            def convert_back_to_image(image_array_str):
                # Convert the string representation of the list back to an actual list
                image_array = ast.literal_eval(image_array_str)
                # Convert the list to a NumPy array and reshape it to the original shape
                return np.array(image_array).reshape(image_height, image_width, num_channels)

            df_X_train['image_array'] = df_X_train['image_array'].apply(convert_back_to_image)
            df_X_test['image_array'] = df_X_test['image_array'].apply(convert_back_to_image)
            
            # Get back the images and labels
            X_train_restored = np.stack(df_X_train['image_array'].values)  # Stack the arrays back into a 4D array
            y_train_restored = df_X_train['label'].values  # Get the labels
            
            X_test_restored=np.stack(df_X_test['image_array'].values)
            y_test_restored=df_X_test['label'].values
            
            # Apply image preprocessing (normalizing)
            # preprocessing_obj = self.get_data_transformation_object()

            logging.info('Applying preprocessing on images')

            # X_train = preprocessing_obj(X_train)
            # X_test = preprocessing_obj(X_test)

            # One-hot encode labels for binary classification (dogs vs cats)
            # y_train = to_categorical(y_train, num_classes=2)
            # y_test = to_categorical(y_test, num_classes=2)

            # Save the preprocessor object
            # save_object(
            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj
            # )

            logging.info('Preprocessing object saved as pickle file')

            # Return the transformed data
            return X_train_restored,y_train_restored,X_test_restored,y_test_restored
        
        except Exception as e:
            logging.error('Exception occurred during data transformation')
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataTransformation()
    train_path = os.path.join(os.getcwd(), 'artifacts', 'train.csv')  # Path to train CSV
    test_path = os.path.join(os.getcwd(), 'artifacts', 'test.csv')    # Path to test CSV
    obj.initiate_data_transformation(train_path, test_path)
