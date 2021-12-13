import base64
import numpy as np
import io
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import re

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)
def get_model():
    global model
    model = load_model('model_beta.h5')
    print("Model Loaded...")
    
def image_process(image, out_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(out_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    
    return image

print(" Loading Keras Model...")
get_model()
    
@app.route('/predict', methods = ['POST'])
def predict():
    # request the image
    msg = request.get_json(force = True)
    # key of json value from msg
    encoded = msg['image']
    img_data = re.sub('^data:image/.+;base64,', '', encoded)
    # decode the encoded image
    decoded = base64.b64decode(img_data)
    
    # image open by wrapping the bytes from the decoded variable
    image = Image.open(io.BytesIO(decoded))
    # process the image using defined method above
    final_img = image_process(image, out_size = (100, 100))
    
    # prediction from the model and convert it to a list
    label = np.argmax(model.predict(final_img))
    pred = model.predict(final_img).tolist()
    if label == 0:
        c = 'Happy'
    elif label == 1:
        c = 'Sad'
    elif label == 2:
        c = 'Fear'
    elif label == 3:
        c = 'Surprise'
    elif label == 4:
        c = 'Neutral'
    elif label == 5:
        c = 'Angry'
    elif label == 6:
        c = 'Disgust'
    
    # store probability of each class for the image
    response = {
        'Prediction': {
            'happy': pred[0][0],
            'sad': pred[0][1],
            'fear': pred[0][2],
            'surprise': pred[0][3],
            'neutral': pred[0][4],
            'angry': pred[0][5],
            'disgust': pred[0][6]
        }
    }
    #print(response)
    return jsonify(response)