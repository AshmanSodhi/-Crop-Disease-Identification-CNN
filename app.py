import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model =load_model('crop-det-cnn.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

image_width = 167
image_height = 250
class_names = ['Healthy', 'Powdery', 'Rust']

def predict(image):
    img = load_img(image,target_size=(image_width,image_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p , cnf = predict(img_path)

	return render_template("index.html", prediction = p, confidence = cnf, img_path = img_path)


if __name__ == '__main__':
    app.run(debug=True)