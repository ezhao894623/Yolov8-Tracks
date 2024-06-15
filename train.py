from ultralytics import YOLO
import os
import torch

# Initialize the YOLO model
print(torch.backends.mps.is_available())
model = YOLO('yolov8n.yaml') #yolo model, n,s,m,l,x

try:
    # Train the model
    model.train(
        data='/path/to/config.yaml',  # Path to the data configuration file
        epochs=800,
        batch=8,
        workers=4,
        lr0=0.001,  # Initial learning rate
        augment=True,
        patience = 0, #prevents the model from ending prematurely when plateauing 
       device = torch.device('cpu')
        
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
