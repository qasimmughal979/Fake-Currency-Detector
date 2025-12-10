
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
from PIL import Image
import io

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Support both local development and Docker deployment
# First try the converted .keras model, then fall back to h5
CONVERTED_MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, '..', 'Module3_Results', 'resnet50_converted.keras'))
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, '..', 'Module3_Results', 'resnet50_frozen_final.h5')

# Determine which model file to use
if os.path.exists(CONVERTED_MODEL_PATH):
    MODEL_PATH = CONVERTED_MODEL_PATH
elif os.path.exists(LEGACY_MODEL_PATH):
    MODEL_PATH = LEGACY_MODEL_PATH
else:
    # Check Docker paths
    docker_model_path = '/app/Module3_Results/resnet50_converted.keras'
    docker_legacy_path = '/app/Module3_Results/resnet50_frozen_final.h5'
    if os.path.exists(docker_model_path):
        MODEL_PATH = docker_model_path
    elif os.path.exists(docker_legacy_path):
        MODEL_PATH = docker_legacy_path
    else:
        MODEL_PATH = CONVERTED_MODEL_PATH  # Will fail, but gives error message

# --- Gemini Vision Setup for Currency Verification ---
from gemini_verifier import GeminiCurrencyVerifier

print("Initializing Gemini Vision for currency verification...")
try:
    gemini_verifier = GeminiCurrencyVerifier()
    print("✓ Gemini Vision verifier ready")
except Exception as e:
    print(f"⚠️ Gemini verifier initialization failed: {e}")
    print("   Set GEMINI_API_KEY environment variable to enable verification")
    gemini_verifier = None
# --------------------------------------------------

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = None

try:
    # Load the model (works for both .keras and .h5 formats)
    import tensorflow.keras as keras
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Use ResNet50's preprocess_input (same as training and test_model.py)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        img_bytes = file.read()
        
        # --- Step 1: Currency Verification with Gemini Vision ---
        verification_note = "Skipped (API key not set)"
        
        if gemini_verifier:
            try:
                verification_result = gemini_verifier.verify_currency(img_bytes)
                is_valid, message = gemini_verifier.get_verification_message(verification_result)
                
                if not is_valid:
                    # Reject if not Pakistani currency
                    return jsonify({'error': message}), 400
                
                verification_note = message
                
            except Exception as e:
                print(f"⚠️ Gemini verification error: {e}")
                verification_note = "Verification failed (proceeding anyway)"
        # -----------------------------------------------

        processed_img = preprocess_image(img_bytes)
        
        # Predict
        prediction = model.predict(processed_img)
        score = float(prediction[0][0])
        
        # Based on training: 0 is Fake, 1 is Real (usually, checking class indices from module3.py is better)
        # In module3.py: {'Fake': 0, 'Real': 1} (Alphabetical order usually, unless specified)
        # Let's assume standard flow_from_directory alphabetical: Fake=0, Real=1
        
        # Sigmoid output: < 0.5 -> Class 0 (Fake), >= 0.5 -> Class 1 (Real)
        
        # Sigmoid output: < 0.5 -> Class 0 (Fake), >= 0.5 -> Class 1 (Real)
        
        label = "Real" if score >= 0.5 else "Fake"
        result_text = "Real Currency Detected" if label == "Real" else "Fake Currency Image Detected"
        confidence = score if score >= 0.5 else 1 - score
        
        return jsonify({
            'result': result_text,
            'label': label,
            'confidence': confidence,
            'raw_score': score,
            'verification': verification_note
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Use 0.0.0.0 to bind to all interfaces (required for Docker)
    # Disable debug mode to prevent watchdog restarts with TensorFlow
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', debug=debug_mode, port=5000, use_reloader=False)
