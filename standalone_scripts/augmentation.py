import os
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# Set the paths
dataset_dir = "archive\hagrid-classification-512p-no-gesture-300k"

# Set the augmentation percentage
augmentation_percentage = 0.3

# Define the augmentations
def augment_image(image):
    augmented_images = []
    
    # Brightness Increase
    enhancer = ImageEnhance.Brightness(image)
    bright_increased = enhancer.enhance(1.5)
    augmented_images.append(bright_increased)
    
    # Brightness Decrease
    bright_decreased = enhancer.enhance(0.5)
    augmented_images.append(bright_decreased)
    
    # Horizontal Flip
    flipped = ImageOps.mirror(image)
    augmented_images.append(flipped)
    
    # Upside Down
    upside_down = ImageOps.flip(image)
    augmented_images.append(upside_down)
    
    return augmented_images

# Function to save the augmented images
def save_images(images, base_path, original_filename):
    base_name, ext = os.path.splitext(original_filename)
    for i, img in enumerate(images):
        augmented_filename = f"{base_name}_aug_{i}{ext}"
        img.save(os.path.join(base_path, augmented_filename))

# Perform data augmentation
for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)
    
    if os.path.isdir(class_path):
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Calculate the number of images to augment
        num_to_augment = int(len(images) * augmentation_percentage)
        images_to_augment = random.sample(images, num_to_augment)
        
        for img_name in images_to_augment:
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            
            # Perform augmentations
            augmented_images = augment_image(img)
            
            # Save the augmented images
            save_images(augmented_images, class_path, img_name)
