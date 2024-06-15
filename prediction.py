import cv2
from PIL import Image
from ultralytics import YOLO
import os
import torch

# Check if MPS is available and use it
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Specify the full path to your model file
model_path = "/path/to/model"

# Specify the task type (e.g., 'detect' for object detection)
task_type = 'detect'

# Load the YOLO model and move it to the appropriate device
try:
    model = YOLO(model_path,task=task_type).to(device)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Predict on images in the 'prediction' directory
try:
    results = model.predict(source="/path/to/photos", show=False, device=device)  # Display predictions
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Define the directory to save the results
output_dir = "/path/to/results"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# Iterate over results and save each image
for i, result in enumerate(results):
    filename = f"result_{i}.jpg" 
    output_path = os.path.join(output_dir, filename)
    print(f"Saving result to {output_path}")
    
    # Plot the results on the image and save
    try:
        annotated_image = result.plot()
        cv2.imwrite(output_path, annotated_image)  # Save the image
    except Exception as e:
        print(f"Error saving result: {e}")
