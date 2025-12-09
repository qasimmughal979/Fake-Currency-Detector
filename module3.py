# ============================================================================
# Module 3: ROBUST Transfer Learning - ResNet50 with Class Imbalance Handling
# ============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ============================================================================
# Configuration
# ============================================================================
DATASET_ROOT = "Final_Clean_Dataset"
OUTPUT_DIR = "Module3_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced for better gradient updates
EPOCHS = 13  # Increased epochs
INITIAL_LR = 0.0001
FINE_TUNE_LR = 0.00001

print("="*70)
print("🚀 Module 3: ROBUST Transfer Learning - ResNet50")
print("="*70)

# ============================================================================
# Focal Loss for Class Imbalance
# ============================================================================
def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for handling class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    
    return focal_loss_fixed

# ============================================================================
# Build Enhanced Model
# ============================================================================
def build_model(num_frozen_layers=140):  # Unfreeze top layers
    print("\n🔧 Loading ResNet50 (ImageNet weights)...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze only bottom layers, allowing top layers to adapt
    for layer in base_model.layers[:num_frozen_layers]:
        layer.trainable = False
    for layer in base_model.layers[num_frozen_layers:]:
        layer.trainable = True
    
    frozen_count = sum([not layer.trainable for layer in base_model.layers])
    trainable_count = sum([layer.trainable for layer in base_model.layers])
    
    print(f"🔒 Frozen layers: {frozen_count}")
    print(f"🔓 Trainable layers: {trainable_count}")
    
    # Enhanced classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    trainable_params = sum([np.prod(w.shape.as_list()) for w in model.trainable_weights])
    total_params = sum([np.prod(w.shape.as_list()) for w in model.weights])
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"📊 Frozen parameters: {total_params - trainable_params:,}")
    
    return model

# ============================================================================
# Data Preparation with Stronger Augmentation
# ============================================================================
print("\n📂 Loading dataset from:", DATASET_ROOT)

if not os.path.exists(DATASET_ROOT):
    print(f"\n❌ ERROR: Dataset folder '{DATASET_ROOT}' not found!")
    exit(1)

# Stronger augmentation for minority class
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_ROOT,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_ROOT,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)
print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}")
print(f"✅ Class mapping: {train_generator.class_indices}")

# ============================================================================
# Calculate Class Weights
# ============================================================================
# Count samples per class
class_counts = {}
for class_name, class_idx in train_generator.class_indices.items():
    class_dir = os.path.join(DATASET_ROOT, class_name)
    count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    class_counts[class_idx] = count

print(f"\n📊 Class distribution:")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"   {class_name}: {class_counts[class_idx]} images")

# Compute class weights to handle imbalance
class_weight_values = compute_class_weight(
    'balanced',
    classes=np.array(list(class_counts.keys())),
    y=np.array(list(class_counts.keys()) * 2)  # Dummy to get balanced weights
)

# Manual calculation for binary
total = sum(class_counts.values())
class_weights = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}

print(f"\n⚖️  Class weights (to balance training):")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"   {class_name}: {class_weights[class_idx]:.2f}")

# ============================================================================
# Train Model
# ============================================================================
model = build_model(num_frozen_layers=140)

# Use focal loss instead of binary crossentropy
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss=focal_loss(gamma=2.0, alpha=0.75),  # Focal loss with higher alpha for minority class
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'resnet50_robust_best.h5'),
        monitor='val_accuracy',  # Changed to monitor accuracy
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

print("\n🚀 Starting training with class weights and focal loss...")
print("="*70)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,  # Apply class weights
    verbose=1
)

# ============================================================================
# Save Final Model
# ============================================================================
final_model_path = os.path.join(OUTPUT_DIR, 'resnet50_robust_final.h5')
model.save(final_model_path)
print(f"\n✅ Final ROBUST model saved: {final_model_path}")

# ============================================================================
# Plot Training History
# ============================================================================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy (With Class Weights)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Focal Loss', fontsize=14, fontweight='bold')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
history_plot_path = os.path.join(OUTPUT_DIR, 'training_history_robust.png')
plt.savefig(history_plot_path, dpi=150)
print(f"✅ Training history saved: {history_plot_path}")

print("\n" + "="*70)
print("🎉 ROBUST TRAINING COMPLETE!")
print("="*70)
print(f"📦 Model: {final_model_path}")
print(f"📊 Improvements:")
print(f"   ✅ Class weights applied (Fake: {class_weights[0]:.2f}, Real: {class_weights[1]:.2f})")
print(f"   ✅ Focal loss for hard examples")
print(f"   ✅ Top {175-140} ResNet layers fine-tuned")
print(f"   ✅ Enhanced augmentation")
print(f"   ✅ Deeper classification head")
print("="*70)
