import os
import glob
import shutil
import random
from pathlib import Path

BASE = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\datasets"

# Define the paths for Traffic Dataset (specifically looking into nested train/val/test)
TRAFFIC_BASE_IMG = os.path.join(BASE, "traffic_vehicles", "Traffic Dataset", "images")
TRAFFIC_BASE_LBL = os.path.join(BASE, "traffic_vehicles", "Traffic Dataset", "labels")

SOURCES = [
    {
        "name": "traffic",
        "images": TRAFFIC_BASE_IMG,
        "labels": TRAFFIC_BASE_LBL,
    },
    {
        "name": "indian_vehicles",
        "images": os.path.join(BASE, "indian_vehicles_yolo", "images"),
        "labels": os.path.join(BASE, "indian_vehicles_yolo", "labels"),
    },
    {
        "name": "ambulance",
        "images": os.path.join(BASE, "ambulance", "ambulance", "images"),
        "labels": os.path.join(BASE, "ambulance", "ambulance", "labels"),
    },
]

OUTPUT_DIR  = os.path.join(BASE, "unified")
VAL_SPLIT   = 0.2
RANDOM_SEED = 42

CLASS_NAMES = [
    "ambulance", "auto", "bicycle", "bus", "car",
    "concrete_mixer", "tempo", "tractor",
    "truck_tanker", "two_wheelers", "vehicle_truck",
]

# Create directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

random.seed(RANDOM_SEED)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
total_train = 0
total_val   = 0

print("\n" + "="*55)
print("  MERGING DATASETS")
print("="*55)

for src in SOURCES:
    name    = src["name"]
    img_dir = src["images"]
    lbl_dir = src["labels"]

    if not os.path.exists(img_dir):
        print(f"\n[SKIP] {name} — images not found: {img_dir}")
        continue

    pairs = []
    
    # RECURSIVE SEARCH for images
    for img_path in glob.glob(os.path.join(img_dir, "**", "*"), recursive=True):
        if Path(img_path).suffix not in IMG_EXTS:
            continue
        
        stem = Path(img_path).stem
        
        # LOGIC FOR NESTED TRAFFIC LABELS: 
        # It checks the root of lbl_dir AND subfolders like 'train' and 'val'
        possible_labels = [
            os.path.join(lbl_dir, stem + ".txt"),
            os.path.join(lbl_dir, "train", stem + ".txt"),
            os.path.join(lbl_dir, "val", stem + ".txt"),
            os.path.join(lbl_dir, "test", stem + ".txt")
        ]
        
        found_lbl = None
        for p_lbl in possible_labels:
            if os.path.exists(p_lbl):
                found_lbl = p_lbl
                break
        
        if found_lbl:
            pairs.append((img_path, found_lbl))

    if not pairs:
        print(f"\n[SKIP] {name} — no image+label pairs found")
        continue

    random.shuffle(pairs)
    n_val       = int(len(pairs) * VAL_SPLIT)
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, lbl_path in split_pairs:
            ext      = Path(img_path).suffix
            stem     = Path(img_path).stem
            new_stem = f"{name}__{stem}"
            shutil.copy2(img_path, os.path.join(OUTPUT_DIR, "images", split, new_stem + ext))
            shutil.copy2(lbl_path, os.path.join(OUTPUT_DIR, "labels", split, new_stem + ".txt"))

    total_train += len(train_pairs)
    total_val   += len(val_pairs)
    print(f"\n  OK {name:20s}  train={len(train_pairs):4d}  val={len(val_pairs):4d}")

# Write data.yaml
yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""# Unified Smart Traffic Dataset
path: {OUTPUT_DIR.replace(chr(92), '/')}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")

print("\n" + "="*55)
print(f"  TOTAL  train={total_train}  val={total_val}")
print(f"  data.yaml -> {yaml_path}")
print("="*55)

if total_train != 0:
    print("\nNext: run train.py")