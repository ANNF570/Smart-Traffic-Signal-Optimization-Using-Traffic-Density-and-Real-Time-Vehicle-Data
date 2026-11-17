"""
FIX + RETRAIN — Remap traffic_vehicles labels to correct 11-class IDs
then rebuild unified dataset and retrain.

Run from:
    F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\

Steps:
    1. Remaps traffic_vehicles labels (0-6) → correct 11-class IDs
    2. Rebuilds unified dataset
    3. Retrains YOLOv8n
"""

import os, glob, shutil, random, torch
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
# TRAFFIC DATASET ORIGINAL CLASS → YOUR 11-CLASS REMAP
# Based on typical traffic-vehicles-object-detection dataset
# Original: 0=two_wheelers, 1=car, 2=auto, 3=bus, 4=truck, 5=vehicle_truck, 6=tempo
# Your IDs:  ambulance=0, auto=1, bicycle=2, bus=3, car=4,
#            concrete_mixer=5, tempo=6, tractor=7, truck_tanker=8,
#            two_wheelers=9, vehicle_truck=10
# ══════════════════════════════════════════════════════════════
TRAFFIC_REMAP = {
    0: 4,   # car           → 4
    1: 9,   # two_wheelers  → 9
    2: 1,   # auto          → 1
    3: 3,   # bus           → 3
    4: 8,   # truck_tanker  → 8
    5: 10,  # vehicle_truck → 10
    6: 6,   # tempo         → 6
}

BASE = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\datasets"

TRAFFIC_LBL_SRC = os.path.join(BASE, "traffic_vehicles", "Traffic Dataset", "labels")
TRAFFIC_LBL_FIX = os.path.join(BASE, "traffic_vehicles_fixed", "labels")
TRAFFIC_IMG_SRC = os.path.join(BASE, "traffic_vehicles", "Traffic Dataset", "images")
TRAFFIC_IMG_FIX = os.path.join(BASE, "traffic_vehicles_fixed", "images")

# ── STEP 1: Remap labels ──────────────────────────────────────
print("="*55)
print("STEP 1 — Remapping traffic_vehicles labels")
print("="*55)

os.makedirs(TRAFFIC_LBL_FIX, exist_ok=True)
os.makedirs(TRAFFIC_IMG_FIX, exist_ok=True)

lbl_files = glob.glob(os.path.join(TRAFFIC_LBL_SRC, "**", "*.txt"), recursive=True)
print(f"Found {len(lbl_files)} label files")

remapped = 0
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

for lbl_path in lbl_files:
    lines = open(lbl_path).read().strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        old_id = int(parts[0])
        new_id = TRAFFIC_REMAP.get(old_id, None)
        if new_id is None:
            continue  # skip unknown classes
        new_lines.append(f"{new_id} " + " ".join(parts[1:]))

    if not new_lines:
        continue

    stem = Path(lbl_path).stem
    out_lbl = os.path.join(TRAFFIC_LBL_FIX, stem + ".txt")
    with open(out_lbl, "w") as f:
        f.write("\n".join(new_lines))

    # Copy matching image
    img_dir = lbl_path.replace("labels", "images").replace(".txt", "")
    img_dir = os.path.dirname(lbl_path).replace("labels", "images")
    for ext in IMG_EXTS:
        src_img = os.path.join(img_dir, stem + ext)
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(TRAFFIC_IMG_FIX, stem + ext))
            break

    remapped += 1

print(f"Remapped: {remapped} label files")

# ── STEP 2: Rebuild unified dataset ──────────────────────────
print("\n" + "="*55)
print("STEP 2 — Rebuilding unified dataset")
print("="*55)

SOURCES = [
    {"name": "traffic",
     "images": TRAFFIC_IMG_FIX,
     "labels": TRAFFIC_LBL_FIX},
    {"name": "indian_vehicles",
     "images": os.path.join(BASE, "indian_vehicles_yolo", "images"),
     "labels": os.path.join(BASE, "indian_vehicles_yolo", "labels")},
    {"name": "ambulance",
     "images": os.path.join(BASE, "ambulance", "ambulance", "images"),
     "labels": os.path.join(BASE, "ambulance", "ambulance", "labels")},
]

OUTPUT_DIR = os.path.join(BASE, "unified")
VAL_SPLIT  = 0.2
random.seed(42)
IMG_EXTS_SET = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

CLASS_NAMES = [
    "ambulance","auto","bicycle","bus","car",
    "concrete_mixer","tempo","tractor",
    "truck_tanker","two_wheelers","vehicle_truck"
]

# Clear old unified dataset
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

for split in ["train","val"]:
    os.makedirs(os.path.join(OUTPUT_DIR,"images",split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR,"labels",split), exist_ok=True)

total_train, total_val = 0, 0

for src in SOURCES:
    name    = src["name"]
    img_dir = src["images"]
    lbl_dir = src["labels"]

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"  [SKIP] {name} — folder not found")
        continue

    pairs = []
    for img_path in glob.glob(os.path.join(img_dir,"**","*"), recursive=True):
        if Path(img_path).suffix not in IMG_EXTS_SET: continue
        stem     = Path(img_path).stem
        lbl_path = os.path.join(lbl_dir, stem+".txt")
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))

    if not pairs:
        print(f"  [SKIP] {name} — no pairs")
        continue

    random.shuffle(pairs)
    n_val       = int(len(pairs)*VAL_SPLIT)
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split, split_pairs in [("train",train_pairs),("val",val_pairs)]:
        for img_path, lbl_path in split_pairs:
            ext      = Path(img_path).suffix
            stem     = Path(img_path).stem
            new_stem = f"{name}__{stem}"
            shutil.copy2(img_path, os.path.join(OUTPUT_DIR,"images",split,new_stem+ext))
            shutil.copy2(lbl_path, os.path.join(OUTPUT_DIR,"labels",split,new_stem+".txt"))

    total_train += len(train_pairs)
    total_val   += len(val_pairs)
    print(f"  OK {name:20s}  train={len(train_pairs):4d}  val={len(val_pairs):4d}")

# Write data.yaml
yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""# Unified Smart Traffic Dataset — remapped
path: {OUTPUT_DIR.replace(chr(92),'/')}
train: images/train
val:   images/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")

print(f"\n  TOTAL  train={total_train}  val={total_val}")
print(f"  data.yaml -> {yaml_path}")

# ── STEP 3: Retrain ───────────────────────────────────────────
print("\n" + "="*55)
print("STEP 3 — Retraining YOLOv8n")
print("="*55)

device = 0 if torch.cuda.is_available() else "cpu"
model  = YOLO("yolov8n.pt")

model.train(
    data     = yaml_path,
    epochs   = 50,
    imgsz    = 416,
    batch    = 4,
    project  = "runs",
    name     = "smart_traffic_fixed",
    exist_ok = True,
    patience = 15,
    workers  = 2,
    device   = device,
    amp      = False,
    augment  = True,
)

print("\nDone! Best weights at:")
print("  runs/detect/runs/smart_traffic_fixed/weights/best.pt")
print("\nUpdate MODEL_NAME in app.py to this path.")