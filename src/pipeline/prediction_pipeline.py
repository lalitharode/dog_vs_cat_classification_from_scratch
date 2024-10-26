import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Path to CNN model file
        self.model_path = os.path.join('artifacts', 'cnn_model.tf')
        print(os.path.exists(self.model_path))

    def load_cnn_model(self):
        """Load the CNN model for image classification."""
        try:
            # Load the trained CNN model
            model = tf.keras.models.load_model(self.model_path)
            logging.info("CNN model loaded successfully.")
            return model
        except Exception as e:
            logging.info("Error loading CNN model.")
            raise CustomException(e, sys)

    def predict(self, image_path):
        """
        Predicts the class of an image using the loaded CNN model.
        Args:
            image_path (str): Path to the image to be predicted.
        Returns:
            Predicted class label.
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not found at the specified path.")

            # Preprocess the image to match CNN input shape (180x180, normalized)
            image = cv2.resize(image, (180, 180))
            image = image / 255.0  # Normalize pixel values
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Load the CNN model
            model = self.load_cnn_model()

            # Perform prediction
            pred = model.predict(image)
            pred_class = np.round(pred).astype(int)[0][0]  # Assuming binary classification
            return "Dog" if pred_class == 1 else "Cat"
        except Exception as e:
            logging.info("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example usage
    image_path = r'c:\Users\lalit\Desktop\dog1.jpg'  # Replace with actual image path
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(image_path)
    print(f"Prediction result: {prediction}")
