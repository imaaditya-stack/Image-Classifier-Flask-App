#Importing neccessary packages and libraries
from __future__ import division, print_function

#Flask
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

#keras
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.mobilenet import preprocess_input, MobileNet

import numpy as np
import os

#Loading MobileNet Pretrained Model
model = MobileNet(include_top=True, weights='imagenet')

model._make_predict_function()

#FLASK INIT
app = Flask(__name__)

#APP CONFIGURATIONS
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['UPLOAD_FOLDER'] = 'images'

app.config['SECRET_KEY'] = '6575fae36288be6d1bad40b99808e37f'

def prepare_image(path):
    """
    This function returns the numpy array of an image
    """
    img = image.load_img(path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return preprocess_input(img_array_expanded_dims)


@app.route('/demo', methods=['GET', 'POST'])

def predict():

    if request.method == 'POST':

        if request.files:

            img = request.files['file']

            extension = img.filename.split('.')[1]

            if extension not in ['jpeg', 'png', 'jpg']:
                flash('File format not suported.', 'warning')

            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(img.filename))

                img.save(file_path)

                preprocessed_image = prepare_image(file_path)

                predictions = model.predict(preprocessed_image)

                labels = decode_predictions(predictions)

                label = labels[0][0]

                label, probablity = label[1], round(label[2]*100,2)

                flash('Predicted class for the input image is {} with probablity {}'.format(label, probablity),'success')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
