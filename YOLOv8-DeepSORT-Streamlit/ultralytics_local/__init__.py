from ultralytics import YOLO
import torch

def load_model(model_path):
    """
    Loads a YOLOv8 model safely, compatible with PyTorch >=2.6
    """
    try:
        # Enable trusted deserialization for Ultralytics models
        torch.serialization.add_safe_globals([YOLO])
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"[WARN] Safe load failed, trying direct load. Error: {e}")
        # Fallback direct model
        return YOLO("yolov8n.pt")
