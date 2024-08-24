import cv2
import mediapipe as mp
import os
import shutil
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = image.shape

        # Get bounding box
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Calculate center and dimensions
        x_center = (x_min + x_max) / (2 * w)
        y_center = (y_min + y_max) / (2 * h)
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        # Prepare data for writing
        data = [f"0 {x_center} {y_center} {width} {height}"]

        # Add all 21 keypoints
        for landmark in hand_landmarks.landmark:
            data.append(f"{landmark.x} {landmark.y} 2")  # 2 is used for visibility as in your example

        return " ".join(data)
    return None

# Input folder containing class folders with images
input_folder = "sampled_dataset"

# Output folders
output_folder = "data"
image_output_folder = os.path.join(output_folder, "images")
label_output_folder = os.path.join(output_folder, "labels")

# Create output folders if they don't exist
for folder in [image_output_folder, label_output_folder]:
    for subfolder in ["train", "val", "test"]:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

# Collect all images
all_images = []
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(os.path.join(root, file))

# Split the data
train_images, temp_images = train_test_split(all_images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

# Process and save the images and labels
def save_data(images, subset):
    for image_path in images:
        result = process_image(image_path)
        if result:
            # Copy the image to the output folder
            output_image_path = os.path.join(image_output_folder, subset, os.path.basename(image_path))
            shutil.copy2(image_path, output_image_path)
            
            # Create and write the label file
            label_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
            output_label_path = os.path.join(label_output_folder, subset, label_file)
            with open(output_label_path, "w") as f:
                f.write(result)

save_data(train_images, "train")
save_data(val_images, "val")
save_data(test_images, "test")

print("Processing complete. Results saved in", output_folder)
