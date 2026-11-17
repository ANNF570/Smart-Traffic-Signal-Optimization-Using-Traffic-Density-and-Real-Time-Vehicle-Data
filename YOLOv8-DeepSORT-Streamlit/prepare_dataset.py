import os
import json
import shutil
import random
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────
BASE_DIR = r"F:\major_project\YOLOv8-DeepSORT-Streamlit"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "yolo_dataset")

SPLITS = ["train", "valid"]

CLASS_NAMES = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
    "car", "garbagevan", "human hauler", "minibus", "minivan",
    "motorbike", "pickup", "policecar", "rickshaw", "scooter",
    "suv", "taxi", "three wheelers -CNG-", "truck", "van", "wheelbarrow"
]
# ──────────────────────────────────────────────────────

def convert_ann_to_yolo(ann_path, img_w, img_h):
    with open(ann_path) as f:
        data = json.load(f)
    
    lines = []
    for obj in data.get("objects", []):
        title = obj["classTitle"]
        if title not in CLASS_NAMES:
            continue
        cls_id = CLASS_NAMES.index(title)
        pts = obj["points"]["exterior"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = abs(x2 - x1) / img_w
        h  = abs(y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

def process_split(split):
    img_dir = os.path.join(DATASET_DIR, split, "img")
    ann_dir = os.path.join(DATASET_DIR, split, "ann")

    out_img = os.path.join(OUTPUT_DIR, "images", split)
    out_lbl = os.path.join(OUTPUT_DIR, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[{split}] Found {len(images)} images")

    converted = 0
    for img_file in images:
        img_path = os.path.join(img_dir, img_file)
        
        # Find annotation file
        ann_file = None
        for ext in [".jpg.json", ".jpeg.json", ".png.json", ".JPG.json", ".PNG.json"]:
            candidate = os.path.join(ann_dir, img_file + ".json")
            if os.path.exists(candidate):
                ann_file = candidate
                break

        if ann_file is None:
            continue

        # Get image dimensions
        from PIL import Image
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        yolo_lines = convert_ann_to_yolo(ann_file, img_w, img_h)

        # Copy image
        shutil.copy(img_path, os.path.join(out_img, img_file))

        # Write label
        stem = Path(img_file).stem
        lbl_path = os.path.join(out_lbl, stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"[{split}] Converted {converted} images ✅")

# ── MAIN ──
print("Starting dataset preparation...")
for split in SPLITS:
    process_split(split)

# Write data.yaml
yaml_content = f"""train: {OUTPUT_DIR}/images/train
val: {OUTPUT_DIR}/images/valid

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n✅ data.yaml created at: {yaml_path}")
print("✅ Dataset preparation complete!")
print(f"\nNow run:\nyolo train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640 project=runs/detect/runs/detect name=my_model exist_ok=True")