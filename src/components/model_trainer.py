# Basic Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
import sys
import os
import cv2


tf.debugging.set_log_device_placement(True)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(os.getcwd(), 'artifacts', 'cnn_model.tf')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_cnn_model(self, input_shape):
        """
        Define a simple CNN model.
        """
        try:
            model = Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(180, 180, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.5),  # Adding Dropout
            layers.Dense(128, activation='relu'),
            layers.Dense(1,activation='sigmoid')
                ])
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            return model

        except Exception as e:
            logging.error("Error in CNN model creation")
            raise CustomException(e, sys)

    def initate_model_training(self, X_train, y_train, X_test, y_test):
        """
        Initiate model training with CNN.
        """
        try:
            logging.info('Starting CNN model training')

            input_shape = X_train.shape[1:]  # Assuming X_train is already reshaped to (batch, height, width, channels)

            # Build the CNN model
            cnn_model = self.build_cnn_model(input_shape)

            # Define early stopping to avoid overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Train the model
            cnn_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=5,
                callbacks=[early_stopping]
            )

            # Evaluate the model on the test set
            loss, accuracy = cnn_model.evaluate(X_test, y_test)
            logging.info(f'CNN Model Test Accuracy: {accuracy}')
            print(f'\n====================================================================================\n')
            print(f'Best Model Found, Accuracy: {accuracy}')
            print(f'\n====================================================================================\n')

            # Save the trained CNN model
            # save_object(self.model_trainer_config.trained_model_file_path,cnn_model)
            cnn_model.save(self.model_trainer_config.trained_model_file_path,save_format='tf')
            logging.info(f'Trained CNN model saved at {self.model_trainer_config.trained_model_file_path}')

        except Exception as e:
            logging.error('Exception occurred during CNN model training')
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Example of loading preprocessed image data (X_train, y_train, X_test, y_test)
    # Ensure that the data is loaded and reshaped properly for CNN input, e.g., (num_samples, height, width, channels)

    # Load your preprocessed data here (this assumes the data is already preprocessed and ready)
    # Replace with actual paths or data loading logic
    try:
        
        obj = DataTransformation()
        train_path = os.path.join(os.getcwd(), 'artifacts', 'train.csv')  # Path to train CSV
        test_path = os.path.join(os.getcwd(), 'artifacts', 'test.csv')    # Path to test CSV
        X_train, y_train, X_test, y_test=obj.initiate_data_transformation(train_path, test_path)

        model = ModelTrainer()
        model.initate_model_training(X_train, y_train, X_test, y_test)
    except Exception as e:
        logging.error("Error loading data or training model")
        raise CustomException(e, sys)
