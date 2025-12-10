
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'Module3_Results')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'resnet50_frozen_final.h5')

# Create directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_dummy_model():
    # Create a simple model with the same input/output signature as the real one
    input_layer = Input(shape=(224, 224, 3))
    x = Flatten()(input_layer)
    # We just want it to run, so a simple dense layer is enough
    # The real model outputs a single value (sigmoid)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=outputs)
    
    # We don't need to compile it for inference if we load with compile=False
    # But let's compile it just in case, using a standard loss
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    print(f"Saving dummy model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    create_dummy_model()
