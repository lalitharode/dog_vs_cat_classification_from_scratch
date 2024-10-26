from flask import Flask, render_template, request, redirect, url_for, flash
from src.pipeline.prediction_pipeline import PredictPipeline
import os

application=Flask(__name__)

app=application 

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Initialize the prediction pipeline and make a prediction
        predictor = PredictPipeline()
        prediction = predictor.predict(file_path)

        # Remove the uploaded file after prediction
        os.remove(file_path)

        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
