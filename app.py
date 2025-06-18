import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import uuid
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploadimages'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and labels
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

with open("plant_disease.json", 'r') as f:
    plant_disease = json.load(f)

@app.route('/uploadimages/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)
    return feature

def predict(image_path):
    features = extract_features(image_path)
    prediction = model.predict(features)
    index = np.argmax(prediction)
    label = labels[index]
    description = plant_disease.get(str(index), "No description available.")
    return label, description

@app.route('/upload/', methods=['POST'])
def upload():
    if 'img' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    image = request.files['img']
    filename = f"{uuid.uuid4().hex}_{image.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    label, description = predict(filepath)

    return jsonify({
        "prediction": label,
        "description": description,
        "image_url": request.url_root + 'uploadimages/' + filename
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Plant Disease Prediction API 🌿"}), 200

if __name__ == '__main__':
    app.run(debug=True)
