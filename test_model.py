#!/usr/bin/env python3
# ============================================================================
# Model Testing Script: ResNet50 Fake Currency Detector
# ============================================================================

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

print("="*70)
print("🧪 Testing ResNet50 Fake Currency Detector")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "Module3_Results/resnet50_frozen_best.h5"
DATASET_ROOT = "Final_Clean_Dataset"
OUTPUT_PATH = "Module3_Results/test_results.png"

# ============================================================================
# Load Model
# ============================================================================
print("\n📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ============================================================================
# Helper Functions
# ============================================================================
def preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_image(image_path):
    """Predict if image is Real or Fake"""
    img, img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Class mapping: {'Fake': 0, 'Real': 1}
    if prediction > 0.5:
        label = "REAL"
        confidence = prediction * 100
    else:
        label = "FAKE"
        confidence = (1 - prediction) * 100
    
    return img, label, confidence

# ============================================================================
# Get Test Images
# ============================================================================
print("\n📂 Loading test images...")

real_folder = os.path.join(DATASET_ROOT, "Real")
fake_folder = os.path.join(DATASET_ROOT, "Fake")

real_images = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
fake_images = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Sample 10 from each class
test_images = random.sample(real_images, 10) + random.sample(fake_images, 10)
random.shuffle(test_images)

print(f"✅ Selected 20 random test images (10 Real + 10 Fake)")

# ============================================================================
# Test and Visualize
# ============================================================================
print("\n🧪 Running predictions...")

results = []
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for idx, img_path in enumerate(test_images):
    # Get ground truth from folder name
    true_label = "REAL" if "Real" in img_path else "FAKE"
    
    # Predict
    img, pred_label, confidence = predict_image(img_path)
    
    # Check if correct
    is_correct = (pred_label == true_label)
    results.append(is_correct)
    
    # Plot
    axes[idx].imshow(img)
    axes[idx].axis('off')
    
    # Color code: green if correct, red if wrong
    color = 'green' if is_correct else 'red'
    
    title = f"True: {true_label}\\nPred: {pred_label} ({confidence:.1f}%)"
    axes[idx].set_title(title, fontsize=10, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"✅ Visualization saved: {OUTPUT_PATH}")

# ============================================================================
# Print Results
# ============================================================================
print("\n" + "="*70)
print("📊 Test Results Summary")
print("="*70)

accuracy = (sum(results) / len(results)) * 100
print(f"✅ Accuracy on 20 random images: {accuracy:.1f}%")
print(f"✅ Correct predictions: {sum(results)}/20")
print(f"❌ Incorrect predictions: {20 - sum(results)}/20")

print("\n" + "="*70)
print("📋 Detailed Results")
print("="*70)

for idx, img_path in enumerate(test_images):
    true_label = "REAL" if "Real" in img_path else "FAKE"
    _, pred_label, confidence = predict_image(img_path)
    is_correct = (pred_label == true_label)
    
    status = "✅" if is_correct else "❌"
    filename = os.path.basename(img_path)
    
    print(f"{status} Image {idx+1:2d}: {filename[:35]:35s} | True: {true_label:4s} | Pred: {pred_label:4s} ({confidence:5.1f}%)")

print("="*70)
print("🎉 Testing complete! Check the visualization at:")
print(f"   {OUTPUT_PATH}")
print("="*70)
