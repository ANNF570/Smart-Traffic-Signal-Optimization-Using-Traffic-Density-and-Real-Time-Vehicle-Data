#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Intelligent Traffic Project
   @Date:          2025/10/21
   @Description:   Configuration file for YOLOv8 + Adaptive Traffic System
-------------------------------------------------
"""
from pathlib import Path
import sys

# -------------------- PATH CONFIGURATION --------------------
# Get the absolute path of the current file
file_path = Path(__file__).resolve()
root_path = file_path.parent

# Add the root path to the sys.path list if not already added
if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path

# -------------------- SOURCE OPTIONS --------------------
SOURCES_LIST = [
    "Image",
    "Video",
    "Webcam"
]

# -------------------- MODEL CONFIGURATION --------------------
# Directory where YOLO models are stored
DETECTION_MODEL_DIR = ROOT / "weights" / "detection"

# List of models available for selection in Streamlit sidebar
DETECTION_MODEL_LIST = [
    "best.pt",          # ✅ Trained ambulance model
    "last.pt",          # Optional previous checkpoint
    "yolov8n.pt",       # Pretrained small YOLOv8
    "yolov8m.pt"        # Pretrained medium YOLOv8
]

# Named paths (for reference)
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"
BEST_MODEL = DETECTION_MODEL_DIR / "best.pt"

# -------------------- OBJECT COUNTER --------------------
OBJECT_COUNTER = None
OBJECT_COUNTER1 = None
