#this is the more accurate version

from ultralytics import YOLO
import os
import torch

# Initialize the YOLO model
print(torch.backends.mps.is_available())
model = YOLO('yolov8x.yaml')

try:
    augmentation_params = {
        'flipud': 0.5,  # Flip images upside down with 50% probability
        'fliplr': 0.5,  # Flip images left-right with 50% probability
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value augmentation
        'rotate': 10.0,  # Rotate images by up to ±10 degrees
        'scale': 0.5,  # Scale images by up to 50%
        'shear': 2.0,  # Shear images by up to ±2 degrees
        'perspective': 0.0,  # Apply perspective transformation with 0% probability
        'mosaic': 1.0,  # Apply mosaic augmentation with 100% probability
        'mixup': 0.0,  # Apply mixup augmentation with 0% probability
    }

    # Train the model
    model.train(
       data='/Users/evanzhao/code/REU/code/data_set/config.yaml',  # Path to the data configuration file
        epochs=800,
        batch=60,
        workers=8, #or how many cores you have
        lr0=0.001,  # Initial learning rate
        augment=True,
        patience = 0, #prevents the model from ending prematurely when plateauing 
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')        
    )
    # Manually set model.ckpt to the model's state dictionary
    model.ckpt = model.model.state_dict()
   
    # Check if the model's state dictionary is not None
    if model.model.state_dict() is not None:
        # Save the trained model
        model.save();
    else:
        print("An error occurred during training: Model state dictionary is None.")
        
except Exception as e:
    print(f"An error occurred during training: {e}")
