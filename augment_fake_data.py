#!/usr/bin/env python3
# ============================================================================
# Data Augmentation Script for Fake Currency Images
# ============================================================================
# This script augments the Fake dataset to balance class distribution

import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

print("="*70)
print("🔄 Augmenting Fake Currency Dataset")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================
FAKE_FOLDER = "Final_Clean_Dataset/Fake"
REAL_FOLDER = "Final_Clean_Dataset/Real"
AUGMENTED_FOLDER = "Final_Clean_Dataset/Fake_Augmented"

# Create backup and augmented folder
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# ============================================================================
# Count existing images
# ============================================================================
fake_images = [f for f in os.listdir(FAKE_FOLDER) 
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
real_images = [f for f in os.listdir(REAL_FOLDER) 
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

num_fake = len(fake_images)
num_real = len(real_images)

print(f"\n📊 Current dataset statistics:")
print(f"   Real images: {num_real}")
print(f"   Fake images: {num_fake}")
print(f"   Ratio: {num_real/num_fake:.2f}:1 (Real:Fake)")

# Calculate how many augmented images we need
target_fake = num_real  # Match the number of real images
augmentations_needed = max(0, target_fake - num_fake)

print(f"\n🎯 Target: {target_fake} fake images")
print(f"   Need to create: {augmentations_needed} augmented images")

if augmentations_needed == 0:
    print("\n✅ Dataset is already balanced!")
    exit(0)

# ============================================================================
# Setup augmentation
# ============================================================================
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# ============================================================================
# Generate augmented images
# ============================================================================
print(f"\n🔧 Generating {augmentations_needed} augmented images...")

augmented_count = 0
images_per_original = int(np.ceil(augmentations_needed / num_fake))

print(f"   Creating ~{images_per_original} augmentation(s) per original image")

for idx, img_file in enumerate(fake_images):
    if augmented_count >= augmentations_needed:
        break
    
    img_path = os.path.join(FAKE_FOLDER, img_file)
    img = Image.open(img_path)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Generate augmented versions
    aug_iter = datagen.flow(img_array, batch_size=1)
    
    for i in range(images_per_original):
        if augmented_count >= augmentations_needed:
            break
            
        aug_img = next(aug_iter)[0].astype('uint8')
        
        # Save to Fake folder with augmented prefix
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        aug_filename = f"aug_{augmented_count}_{base_name}{ext}"
        aug_path = os.path.join(FAKE_FOLDER, aug_filename)
        
        Image.fromarray(aug_img).save(aug_path)
        augmented_count += 1
        
        if (augmented_count + 1) % 100 == 0:
            print(f"   Generated {augmented_count}/{augmentations_needed} images...")

print(f"✅ Generated {augmented_count} augmented images")

# ============================================================================
# Verify final counts
# ============================================================================
final_fake_images = [f for f in os.listdir(FAKE_FOLDER) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print("\n" + "="*70)
print("📊 Final dataset statistics:")
print("="*70)
print(f"   Real images: {num_real}")
print(f"   Fake images: {len(final_fake_images)} (original: {num_fake}, augmented: {augmented_count})")
print(f"   New ratio: {num_real/len(final_fake_images):.2f}:1 (Real:Fake)")
print("="*70)
print("✅ Data augmentation complete!")
print("="*70)
