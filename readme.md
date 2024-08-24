# Hand Keypoint Detection and Classification

This project demonstrates a hand gesture recognition system using a two-stage inference pipeline. The system consists of a YOLOv8 model for keypoint detection and a CNN classifier model for gesture classification. The models are quantized to improve performance on edge devices.


## Table of Contents
* [Introduction](#introduction)
* [Models Used](#models-used)
* [Quantization](#quantization)
* [Inference Pipeline](#inference-pipeline)
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

## Introduction

The project aims to recognize hand gestures by first detecting keypoints using a YOLOv8 model and then classifying the gesture using a CNN model. The models are quantized to reduce the computational load and make them suitable for deployment on devices with limited resources.


## Models Used

1. YOLOv8: Fine-tuned for hand keypoint detection.
2. CNN Classifier: Trained to classify hand gestures based on the detected keypoints.

## Quantization

- YOLOv8: Quantized to FP16 using OpenVINO for optimized inference.
- CNN Classifier: Quantized to INT8 using TensorFlow Lite for efficient classification on edge device

## Inference Pipeline

- YOLOv8: The model takes input frames and detects hand keypoints along with bounding boxes.
- Cropping: The detected hand is cropped from the frame using the bounding box.
- CNN Classifier: The cropped hand image is passed to the classifier model to determine the gesture.

## Installation

1. Clone the repository:

- `git clone https://github.com/your-username/hand_key_point_detection_finetuning_with_image_classification`
- `cd hand-keypoint-detection-classification`

2. Set up the virtual environment:
- `python3 -m venv venv`
- `source venv/bin/activate` # for linux 
- `venv\Scripts\activate` # for windows

3. Install the required dependencies:
- `pip install -r requirements.txt`

## Usage

- To run inference using the quantized models, use
`python inference.py`

## Results

- Quantization significantly reduces model size and inference time with minimal impact on accuracy, making the models suitable for deployment on edge devices.

