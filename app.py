from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import os
import boto3
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


import requests
from io import BytesIO


# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image





import sys











# Load environment variables
load_dotenv()



# Define a flask app
app = Flask(__name__)





# S3 Configuration
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
model_key = 'my_model.h5'  # Replace this with your actual model path in S3
download_path = 'my_model.h5'  # Local path where the model will be downloaded

def upload_file_to_s3(file, bucket_name, object_name):
    try:
        s3.upload_fileobj(file, bucket_name, object_name)
        return f"https://{bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{object_name}"
    except Exception as e:
        print(f"Something happened: {e}")
        return None



# Define constants
IMAGE_SIZE = 128
NUM_CLASSES = 4  # Assuming two classes: Tumor and NoTumor

# Function to download model from S3
def download_model(bucket_name, model_key, download_path):
    try:
        s3.download_file(bucket_name, model_key, download_path)
        print(f"Model downloaded successfully from S3 to {download_path}.")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Check if model exists locally; if not, download it
def load_or_download_model():
    model_path = "my_model.h5"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Downloading from S3...")
        download_model(bucket_name, model_key, model_path)
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        print("****************************************************************")
        print("NICE")
        print("****************************************************************")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

# Load the model during app startup
model = load_or_download_model()


# Function to predict on a single image
def predict_image(image_path, model):
    try:
        # Download the image from the S3 URL
        response = requests.get(image_path)
        print(f"Image download status: {response.status_code}")
        # Load and preprocess the image
        img = Image.open(BytesIO(response.content)).convert('RGB')  # Convert to RGB to ensure 3 channels
        x = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))

        # Ensure the image has the correct shape and channels
        if len(x.shape) == 2:  # If the image is grayscale, convert it to RGB format
            x = np.stack((x,) * 3, axis=-1)
        
        x = x.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        x = x / 255.0  # Normalizing the image
        # Perform the prediction
        res = model.predict_on_batch(x)
        return res
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    # classification = np.argmax(res, axis=-1)[0]

    # # Output the prediction
    # labels = ["Glioma", "Meningioma", "NO Tumor", "Pituitary"]
    # print(f"{res[0][classification] * 100:.2f}% Conclusion: {labels[classification]}")


# Example usage
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\meningioma\Te-me_0097.jpg")
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\pituitary\Te-pi_0013.jpg")
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\notumor\Te-no_0013.jpg")
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\glioma\Te-gl_0013.jpg")

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        
        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            s3_url = upload_file_to_s3(file, bucket_name, filename)
        print("********************************************************************************")
        print(s3_url)
        print("********************************************************************************")

        # Make prediction
        print("Before Predictiono")
        preds = predict_image(s3_url, model)
        print("After Predictiono")
        classification = np.argmax(preds, axis=-1)[0]

        labels = ["Glioma", "Meningioma", "NO Tumor", "Pituitary"]
        return (f"{preds[0][classification] * 100:.2f}% Conclusion: {labels[classification]}")
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # #**************************************************************************************************
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result
        #*************************************************************************************************

    return None


if __name__ == '__main__':
    app.run(debug=True)
