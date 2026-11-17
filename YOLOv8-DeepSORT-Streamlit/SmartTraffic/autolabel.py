"""
STEP 2 — Auto-label ambulance images using pretrained YOLOv8
Reads from: datasets/ambulance/ambulance/images/
Writes to:  datasets/ambulance/ambulance/labels/

Uses COCO pretrained model — treats car/bus/truck as ambulance proxy.
IMPORTANT: Spot-check labels in LabelImg before training.
"""

import os
import glob
from pathlib import Path
from ultralytics import YOLO

BASE           = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\datasets"
IMAGES_DIR     = os.path.join(BASE, "ambulance", "ambulance", "images")
LABELS_DIR     = os.path.join(BASE, "ambulance", "ambulance", "labels")

os.makedirs(LABELS_DIR, exist_ok=True)

# COCO pretrained — car=2, bus=5, truck=7 used as ambulance proxies
model         = YOLO("yolov8x.pt")
PROXY_CLASSES = {2, 5, 7}
CONF          = 0.35

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    image_paths += glob.glob(os.path.join(IMAGES_DIR, "**", ext), recursive=True)

print(f"Found {len(image_paths)} ambulance images")
labeled, empty = 0, 0

for img_path in image_paths:
    results = model(img_path, conf=CONF, verbose=False)[0]
    lines   = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in PROXY_CLASSES:
            continue
        x_center, y_center, bw, bh = box.xywhn[0].tolist()
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    stem       = Path(img_path).stem
    label_file = os.path.join(LABELS_DIR, stem + ".txt")

    with open(label_file, "w") as f:
        f.write("\n".join(lines))

    if lines:
        labeled += 1
    else:
        empty += 1

print(f"Labeled : {labeled}")
print(f"Empty   : {empty}  (no vehicle detected)")
print(f"Labels  : {LABELS_DIR}")
print("\nNext: run step3_merge_dataset.py")