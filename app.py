# Import the necessary libraries
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from flask import Flask, request, render_template, jsonify

# Load the saved model
model_path = 'model.p'
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Initialize the Flask app
app = Flask(__name__)


# Function to preprocess and predict a single image
def predict_image(image_path, model):
    # Read and preprocess the image
    img = imread(image_path)
    img = resize(img, (15, 15)).flatten()

    # Make a prediction using the loaded model
    prediction = model.predict([img])
    return prediction


# Define a route to display the upload form
@app.route('/')
def index():
    return render_template('upload.html')


# Define a route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', error_message="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', error_message="No selected file")

    if file:
        # Save the uploaded file to a temporary location
        uploaded_image_path = 'test.jpg'
        file.save(uploaded_image_path)

        # Predict the category of the uploaded image
        predicted_label = predict_image(uploaded_image_path, loaded_model)

        # Check if the prediction confidence is less than 50%
        if max(predicted_label) < 0.5:
            return render_template('error.html', error_message="Opps not mode for this ... ")

        # Decode the predicted label
        categories = ['healthy', 'disease']
        predicted_category = categories[predicted_label[0]]

        # Pass the prediction to the result page
        return render_template('result.html', prediction=predicted_category)


if __name__ == '__main__':
    app.run(debug=True)
