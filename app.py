import os
import uuid
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import gdown

# --- CONFIGURATION ---
MODEL_PATH = "models/plant_disease_model.keras"  # Changed from .h5 to .keras
DRIVE_FILE_ID = "1qDqeP1rHcawATIR4sv3WRULHJUh-FUFO"
UPLOAD_FOLDER = "uploadimages"
LABELS_PATH = "plant_disease.json"

# --- DOWNLOAD MODEL SAFELY ---
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model from Google Drive...")
    os.makedirs("models", exist_ok=True)
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # Validate if the model is an actual Keras file
    if os.path.getsize(MODEL_PATH) < 1000000:  # ~1MB is suspicious
        raise ValueError("Downloaded file is too small. Check Google Drive file format and permissions.")
else:
    print("✅ Model already exists. Skipping download.")

# --- APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD MODEL & LABELS ---
print("🔁 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", str(e))
    raise

# --- LOAD LABELS ---
with open(LABELS_PATH, 'r') as f:
    plant_disease = json.load(f)

labels = list(plant_disease.values())

# --- UTILITIES ---
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path):
    features = extract_features(image_path)
    prediction = model.predict(features)
    index = int(np.argmax(prediction))
    label = labels[index] if index < len(labels) else "Unknown"
    description = plant_disease.get(str(index), "No description available.")
    return label, description

# --- ROUTES ---
@app.route('/')
def home():
    return jsonify({"message": "🌿 Plant Disease Detection API running successfully"}), 200

@app.route('/upload/', methods=['POST'])
def upload_image():
    if 'img' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

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

@app.route('/uploadimages/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --- MAIN ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses PORT env
    app.run(host="0.0.0.0", port=port)
