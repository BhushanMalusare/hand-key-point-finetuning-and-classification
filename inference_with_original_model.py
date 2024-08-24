import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# Load YOLO model
yolo_model = YOLO("B:\hand_key_point_finetuned\quantized_models\last_openvino_model", task='pose')

# Load the original TensorFlow model
classifier_model = tf.keras.models.load_model("B:\hand_key_point_finetuned\models\classifier.h5")

class_names = ['ac_off', 'ac_on', 'fan_off', 'fan_on', 'light_off', 'light_on', 'switch_off', 'switch_on']

# Function to preprocess image for TensorFlow model
def preprocess_image(image, input_size):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to get class prediction
def get_class_prediction(image):
    prediction = classifier_model.predict(image)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    
    confidence_threshold = 0.7  # Adjust this value as needed
    # if confidence > confidence_threshold:
    return class_names[class_id], confidence
    # else:
        # return None, confidence

# Parameters
yolo_conf_threshold = 0.7
keypoint_conf_threshold = 0.6
classifier_conf_threshold = 0.0

# Manually set the input size for the classifier
input_size = (256, 256)  # Replace with your model's actual input size

# Open video capture (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get YOLO results
    results = yolo_model(frame, conf=yolo_conf_threshold, iou=0.7)
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        for i in range(len(boxes)):
            # Get bounding box
            box = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # Crop the bounding box from the original image
            cropped_image = frame[y1:y2, x1:x2]

            # Get class prediction for the cropped image
            preprocessed_image = preprocess_image(cropped_image, input_size)
            predicted_class, confidence = get_class_prediction(preprocessed_image)

            # Draw keypoints and class text
            kps = keypoints.xy[i].cpu().numpy()
            kps_conf = keypoints.conf[i].cpu().numpy() if keypoints.conf is not None else None
            for j, (x, y) in enumerate(kps):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if kps_conf is None or kps_conf[j] > keypoint_conf_threshold:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Add predicted class text to the image
            if predicted_class is not None and confidence > classifier_conf_threshold:
                cv2.putText(frame, f"Class: {predicted_class} ({confidence:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # else:
                # cv2.putText(frame, "No confident prediction", 
                            # (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Hand Keypoints and Class", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
