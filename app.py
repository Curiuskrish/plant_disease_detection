import os
import uuid
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import gdown

# --- CONFIGURATION ---
MODEL_PATH = "plant_disease_model.h5"  # Flat structure - model in root
DRIVE_FILE_ID = "1qDqeP1rHcawATIR4sv3WRULHJUh-FUFO"
UPLOAD_FOLDER = "uploadimages"
LABELS_PATH = "plant_disease.json"

# --- DOWNLOAD MODEL IF NOT EXISTS ---
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    if os.path.getsize(MODEL_PATH) < 1_000_000:
        raise ValueError("❌ Downloaded file too small. Likely not a valid model. Check permissions.")
else:
    print("✅ Model already exists.")

# --- APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD MODEL ---
print("🔁 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    raise

# --- LOAD LABELS ---
try:
    with open(LABELS_PATH, 'r') as f:
        plant_disease = json.load(f)
    print("✅ Labels loaded.")
except Exception as e:
    print("❌ Failed to load labels:", str(e))
    raise

labels = list(plant_disease.values())

# --- UTILITIES ---
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(img_array, axis=0)

def predict(image_path):
    features = extract_features(image_path)
    prediction = model.predict(features)
    index = int(np.argmax(prediction))
    label = labels[index] if index < len(labels) else "Unknown"

    description = plant_disease.get(str(index), {
        "name": label,
        "cause": "Unknown cause",
        "cure": "No cure information available."
    })

    return description

# --- ROUTES ---
@app.route('/')
def home():
    return jsonify({"message": "🌿 Plant Disease Detection API is running."}), 200

@app.route('/upload/', methods=['POST'])
def upload_image():
    if 'img' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    image = request.files['img']
    filename = f"{uuid.uuid4().hex}_{image.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    prediction = predict(filepath)

    return jsonify({
        "prediction": prediction['name'],
        "cause": prediction['cause'],
        "cure": prediction['cure'],
        "image_url": request.url_root + 'uploadimages/' + filename
    })

@app.route('/uploadimages/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --- MAIN ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
