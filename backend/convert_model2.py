"""
Script to properly extract and load weights from legacy Keras 2.x h5 model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py

MODEL_PATH = "../Module3_Results/resnet50_frozen_final.h5"

print("=== Extracting weights from h5 file ===")

# Extract weights from h5 file
with h5py.File(MODEL_PATH, 'r') as f:
    print("\nDense layer weights structure:")
    
    # Get dense layer weights
    def get_weights(f, layer_path):
        """Extract weights from a layer in h5 file"""
        weights = []
        if layer_path in f['model_weights']:
            layer_group = f['model_weights'][layer_path]
            # Navigate to the actual weights
            for key in layer_group.keys():
                if isinstance(layer_group[key], h5py.Group):
                    for key2 in layer_group[key].keys():
                        if isinstance(layer_group[key][key2], h5py.Group):
                            for key3 in layer_group[key][key2].keys():
                                data = layer_group[key][key2][key3][:]
                                print(f"  {layer_path}/{key}/{key2}/{key3}: shape {data.shape}")
                                weights.append(data)
        return weights
    
    # Check dense layers
    dense_weights = get_weights(f, 'dense')
    dense_1_weights = get_weights(f, 'dense_1')
    dense_2_weights = get_weights(f, 'dense_2')

print("\n=== Building model and loading weights manually ===")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Build the architecture - use imagenet weights for resnet base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', name='dense'),
    layers.Dropout(0.5, name='dropout'),
    layers.Dense(128, activation='relu', name='dense_1'),
    layers.Dropout(0.3, name='dropout_1'),
    layers.Dense(1, activation='sigmoid', name='dense_2')
])

# Build the model by running a forward pass
model.build((None, 224, 224, 3))

# Now load the weights from h5 file
print("\nLoading weights from h5 file...")
with h5py.File(MODEL_PATH, 'r') as f:
    # Load dense layer weights
    def load_layer_weights(model, layer_name, h5_layer_path, f):
        """Load weights for a specific layer from h5 file"""
        layer = model.get_layer(layer_name)
        
        # Navigate to weights
        layer_group = f['model_weights'][h5_layer_path]
        
        # Get kernel and bias
        kernel_data = None
        bias_data = None
        
        # Find the actual weight data
        def find_weights(group, weights_list):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    weights_list.append((key, item[:]))
                elif isinstance(item, h5py.Group):
                    find_weights(item, weights_list)
        
        weights_list = []
        find_weights(layer_group, weights_list)
        
        # Sort by name (bias, kernel)
        weights_list = sorted(weights_list, key=lambda x: x[0], reverse=True)  # kernel before bias
        
        if len(weights_list) == 2:
            # kernel:0 and bias:0
            kernel = [w[1] for w in weights_list if 'kernel' in w[0]][0]
            bias = [w[1] for w in weights_list if 'bias' in w[0]][0]
            layer.set_weights([kernel, bias])
            print(f"  Loaded {layer_name}: kernel {kernel.shape}, bias {bias.shape}")
            return True
        else:
            print(f"  Warning: Expected 2 weights for {layer_name}, found {len(weights_list)}")
            return False
    
    # Load ResNet50 weights from the saved model (fine-tuned weights)
    resnet_layer = model.get_layer('resnet50')
    resnet_group = f['model_weights']['resnet50']
    
    # Load weights for each layer in resnet50
    loaded_count = 0
    for layer in resnet_layer.layers:
        if len(layer.weights) > 0:
            layer_name = layer.name
            if layer_name in resnet_group:
                weights_list = []
                find_weights_group = resnet_group[layer_name]
                
                def collect_weights(group, weights_list):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            weights_list.append((key, item[:]))
                        elif isinstance(item, h5py.Group):
                            collect_weights(item, weights_list)
                
                collect_weights(find_weights_group, weights_list)
                
                if len(weights_list) > 0:
                    # Sort weights to match layer.weights order
                    weight_names = [w.name.split('/')[-1].replace(':0', '') for w in layer.weights]
                    sorted_weights = []
                    for wn in weight_names:
                        for name, data in weights_list:
                            if wn in name:
                                sorted_weights.append(data)
                                break
                    
                    if len(sorted_weights) == len(layer.weights):
                        try:
                            layer.set_weights(sorted_weights)
                            loaded_count += 1
                        except Exception as e:
                            pass  # Skip mismatched shapes

    print(f"  Loaded {loaded_count} ResNet50 layer weights")
    
    # Load dense layer weights
    load_layer_weights(model, 'dense', 'dense', f)
    load_layer_weights(model, 'dense_1', 'dense_1', f)
    load_layer_weights(model, 'dense_2', 'dense_2', f)

print("\n=== Testing model inference ===")

# Create a test image
test_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
test_img = preprocess_input(test_img)

output = model.predict(test_img, verbose=0)
print(f"Test output: {output}")

if not np.isnan(output[0][0]):
    print("✓ Model inference works!")
    
    # Save the model in new format
    output_path = "../Module3_Results/resnet50_converted.keras"
    model.save(output_path)
    print(f"✓ Model saved to: {output_path}")
    
    # Also save weights only
    weights_path = "../Module3_Results/resnet50_weights.weights.h5"
    model.save_weights(weights_path)
    print(f"✓ Weights saved to: {weights_path}")
else:
    print("✗ Model output is NaN - weights may not have loaded correctly")
