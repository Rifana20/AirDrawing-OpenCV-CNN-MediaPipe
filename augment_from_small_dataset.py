import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Source and target folders
INPUT_DIR = "images"  # Contains 10 images per class
OUTPUT_DIR = "augmented_images"  # Will store 100 images per class

# Number of augmentations per image (10 original * 10 = 100 total)
AUG_PER_IMAGE = 10

# Set up augmentation generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through all classes
for class_name in tqdm(os.listdir(INPUT_DIR), desc="Augmenting"):
    class_input_path = os.path.join(INPUT_DIR, class_name)
    class_output_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    for img_name in os.listdir(class_input_path):
        img_path = os.path.join(class_input_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Failed to load {img_path}")
            continue

        img = cv2.resize(img, (64, 64))
        img = img.reshape((1, 64, 64, 1))

        i = 0
        for batch in datagen.flow(img, batch_size=1):
            save_path = os.path.join(class_output_path, f"{class_name}_{img_name.split('.')[0]}_{i}.jpg")
            cv2.imwrite(save_path, batch[0].reshape(64, 64))
            i += 1
            if i >= AUG_PER_IMAGE:
                break

print("✅ Augmentation complete. Check 'augmented_images/' folder.")
