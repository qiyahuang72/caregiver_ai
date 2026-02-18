import torch
from ultralytics import YOLO

# 1. Check if PyTorch sees the GPU
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Test YOLOv8-Pose loading
try:
    model = YOLO('yolov8n-pose.pt') # Downloads the tiny model for testing
    print("YOLOv8-Pose model loaded successfully!")
    # Move to GPU
    model.to('cuda')
    print("Model successfully moved to GB10!")
except Exception as e:
    print(f"Error: {e}")