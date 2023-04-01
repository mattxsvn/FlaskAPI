from flask import Flask, request, Response, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import normalize
import numpy as np
import random as rn
import os
import io

sd = 1 # Here sd means seed.
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED']=str(sd)
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
import cv2
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model('C:/Users/mattg/OneDrive/Documents/vscode/epoch100_Unet.h5', compile=False)


# Define a route for your API
@app.route('/predict', methods=['POST'])

def predict():

    responses = []

    # Loop through all the images in the request
    for img_file in request.files.getlist('image'):
        # Get the image from the request
        img_file = img_file.read()

        # Convert the image data to a numpy array
        npimg = np.frombuffer(img_file, np.uint8)
        
        # Decode the numpy array to an OpenCV image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        img,img_bgr = preprocess_image(img)
        
        # Make a prediction using your model
        pred = model.predict(img)
        
        # Postprocess the prediction
        output_img = postprocess_output(pred)
        img_bgr[output_img==2] = 255
        # Encode the image as JPEG
        ret, jpeg = cv2.imencode('.jpg', img_bgr)

        #Calculate for the pixel percentage of disease to fruit
        disease_percentage = pixel_calculations(output_img)

        #Get the severity level
        severity_level = determine_severity(disease_percentage)

        # Create a Flask response with the encoded image
        #response = Response(jpeg.tobytes(), mimetype='imagjsonify(response)

        responses.append({
                "image": jpeg.tobytes().decode('latin-1'),
                "disease_percentage": disease_percentage,
                "severity_level": severity_level
        })
    return jsonify(responses)

def preprocess_image(img):
    # Preprocess the image here
    test_images = [] #(1,128,128,3)
    SIZE_Y = SIZE_X = 256
    BACKBONE1 = 'resnet34'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    test_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_images.append(test_image)
    test_images = np.array(test_images)
    test_images = normalize(test_images, axis=1)
    test = preprocess_input1(test_images)
    return test,img

def postprocess_output(pred):
    # Postprocess the prediction here
    predicted_img=np.argmax(pred, axis=3)[0,:,:]
    return predicted_img

def pixel_calculations(img):
    # Get the number of pixels in each color channel
    healthy_pixels = np.count_nonzero(img == 1)
    disease_pixels = np.count_nonzero(img == 2)
    # Get total pixels for whole fruit
    fruit_pixels = healthy_pixels + disease_pixels
    # Get the percentage
    dss_prcntg =  round((disease_pixels / fruit_pixels) * 100, 2)
    return dss_prcntg

def determine_severity(prcntg):
    if prcntg == 0:
        level = 0
    elif 0 < prcntg <= 10:
        level = 1
    elif 10 < prcntg <= 25:
        level = 2
    elif 25 < prcntg <= 50:
        level = 3
    elif 50 < prcntg <= 75:
        level = 4
    else:
        level = 5
    return level

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5000, debug=True)