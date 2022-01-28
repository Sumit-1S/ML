from __future__ import division, print_function
# coding=utf-8
import sys
import os
from glob import glob
import re
import numpy as np
# import Image
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')
folders = glob('Datasets\*')

def model_predict(img_path, model):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(150,150))


    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    print(preds)
    return preds
    
    # image = Image.open(file)
    # st.image(image,use_column_width=True)
    # prediction = import_and_predict(image,model)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('idx.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        if preds[0][np.argmax(preds)]>0.97:
            result = folders[np.argmax(preds)][9:]# Convert to string
        else:
            result = "No Match Found"
        print(np.argmax(preds))
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)