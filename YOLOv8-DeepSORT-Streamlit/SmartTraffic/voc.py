"""
STEP 1 — Convert Indian Vehicles Pascal VOC XML -> YOLO .txt
Reads from: datasets/indian_vehicles/Indian_vehicle_dataset/
Writes to:  datasets/indian_vehicles_yolo/images/ + labels/
"""

import os
import glob
import shutil
from xml.etree import ElementTree as ET
from pathlib import Path

BASE       = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\datasets"
INPUT_DIR  = os.path.join(BASE, "indian_vehicles", "Indian_vehicle_dataset")
OUTPUT_DIR = os.path.join(BASE, "indian_vehicles_yolo")

CLASS_NAMES = [
    "ambulance",      # 0
    "auto",           # 1
    "bicycle",        # 2
    "bus",            # 3
    "car",            # 4
    "concrete_mixer", # 5
    "tempo",          # 6
    "tractor",        # 7
    "truck_tanker",   # 8
    "two_wheelers",   # 9
    "vehicle_truck",  # 10
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print(f"Scanning: {INPUT_DIR}")
xml_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.xml"), recursive=True)
print(f"Found {len(xml_files)} XML files\n")

IMG_EXTS   = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
converted  = 0
skipped    = 0
unknown_cls = set()

for xml_path in xml_files:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size   = root.find("size")
        img_w  = int(size.find("width").text)
        img_h  = int(size.find("height").text)

        if img_w == 0 or img_h == 0:
            skipped += 1
            continue

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip().lower()
            if cls_name not in CLASS_MAP:
                unknown_cls.add(cls_name)
                continue
            cls_id = CLASS_MAP[cls_name]
            bndbox = obj.find("bndbox")
            xmin   = float(bndbox.find("xmin").text)
            ymin   = float(bndbox.find("ymin").text)
            xmax   = float(bndbox.find("xmax").text)
            ymax   = float(bndbox.find("ymax").text)
            cx     = (xmin + xmax) / 2 / img_w
            cy     = (ymin + ymax) / 2 / img_h
            bw     = (xmax - xmin) / img_w
            bh     = (ymax - ymin) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            skipped += 1
            continue

        stem       = Path(xml_path).stem
        img_dir    = os.path.dirname(xml_path)
        label_path = os.path.join(OUTPUT_DIR, "labels", stem + ".txt")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        # Copy matching image
        for ext in IMG_EXTS:
            src_img = os.path.join(img_dir, stem + ext)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(OUTPUT_DIR, "images", stem + ext))
                break

        converted += 1

    except Exception as e:
        print(f"  [ERROR] {xml_path}: {e}")
        skipped += 1

print(f"Converted : {converted}")
print(f"Skipped   : {skipped}")
if unknown_cls:
    print(f"Unknown classes (not added): {unknown_cls}")
print(f"\nOutput: {OUTPUT_DIR}")
print("Next: run step2_autolabel_ambulance.py")