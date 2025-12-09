
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
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
MODEL_PATH = os.path.join(BASE_DIR, '..', 'Module3_Results', 'resnet50_robust_best.h5')

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

# Define Focal Loss (Required for loading the model)
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed

# Load Model
print(f"Loading model from {MODEL_PATH}...")
try:
    # We need to register the custom loss function
    # Note: The model might have been saved with specific parameters for focal_loss,
    # but when loading with custom_objects, providing the function factory or the function itself usually works.
    # If the model was saved with the *result* of the factory, we might need to recreate it exactly or use a wrapper.
    # However, 'focal_loss' is the name likely stored. The function defined in module3 returns 'focal_loss_fixed'.
    # Let's try mapping 'focal_loss_fixed' to the inner function.
    
    # Actually, based on module3.py: loss=focal_loss(...)
    # Keras saves the function name. The inner function is 'focal_loss_fixed'.
    
    # To be safe, we load without compiling first to avoid loss mismatch issues if we are just predicting.
    model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()}, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize as per training
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
        
        result = "Real" if score >= 0.5 else "Fake"
        confidence = score if score >= 0.5 else 1 - score
        
        return jsonify({
            'result': result,
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
    app.run(debug=True, port=5000)
