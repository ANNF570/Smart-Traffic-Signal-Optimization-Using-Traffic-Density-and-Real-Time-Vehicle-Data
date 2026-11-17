"""
=============================================================
  TRAFFIC VEHICLE DETECTION - MERGE + TRAIN SCRIPT v2
  Project: F:\\major_project\\YOLOv8-DeepSORT-Streamlit
  Run: python merge_and_train.py
  (Corrected remap for all discovered class names)
=============================================================
"""

import os
import shutil
import yaml
import torch
from collections import defaultdict

BASE = r"F:\major_project\YOLOv8-DeepSORT-Streamlit"
OUT  = os.path.join(BASE, "merged_dataset")

FINAL_CLASSES = [
    "ambulance",    # 0
    "auto",         # 1
    "bicycle",      # 2
    "bike",         # 3
    "bus",          # 4
    "car",          # 5
    "motorcycle",   # 6
    "truck",        # 7
    "van",          # 8
    "other",        # 9
]

REMAP = {
    # ambulance
    "ambulance": 0,
    # auto / rickshaw / three-wheeler
    "auto": 1, "autorickshaw": 1, "auto-rickshaw": 1,
    "auto rickshaw": 1, "rickshaw": 1, "cng": 1,
    "three-wheeler": 1, "threewheeler": 1, "threewheel": 1,
    "three wheeler": 1, "vikram": 1,
    # bicycle
    "bicycle": 2, "cycle": 2,
    # bike / scooter / two-wheeler
    "bike": 3, "scooter": 3, "two-wheeler": 3,
    "twowheeler": 3, "two wheeler": 3,
    # bus
    "bus": 4, "minibus": 4,
    # car
    "car": 5, "hatchback": 5, "sedan": 5, "suv": 5,
    "police-car": 5, "police car": 5, "policecar": 5,
    "taxi": 5, "cab": 5, "modified": 5,
    # motorcycle
    "motorcycle": 6, "motorbike": 6, "motor bike": 6,
    # truck
    "truck": 7, "lorry": 7, "firetruck": 7, "fire truck": 7,
    "fire-truck": 7, "tractor": 7, "pickup": 7,
    "heavy vehicle": 7, "heavyvehicle": 7,
    # van
    "van": 8, "minivan": 8, "tempo": 8,
    # other / catch-all
    "vehicle": 9, "other": 9, "unknown": 9,
    "person": 9, "pedestrian": 9,
    "fixed obstacle": 9, "fixed_obstacle": 9, "obstacle": 9,
}

SOURCES = [
    {
        "name": "emergency_vehicles",
        "yaml": os.path.join(BASE, "Emergency vehicles.v1-emergency-vehicle.yolov8", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "Emergency vehicles.v1-emergency-vehicle.yolov8", "train", "images"),
                os.path.join(BASE, "Emergency vehicles.v1-emergency-vehicle.yolov8", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "Emergency vehicles.v1-emergency-vehicle.yolov8", "valid", "images"),
                os.path.join(BASE, "Emergency vehicles.v1-emergency-vehicle.yolov8", "valid", "labels"),
            ),
        }
    },
    {
        "name": "indian_traffic_objects",
        "yaml": os.path.join(BASE, "Indian traffic objects.v1i.yolov8", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "Indian traffic objects.v1i.yolov8", "train", "images"),
                os.path.join(BASE, "Indian traffic objects.v1i.yolov8", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "Indian traffic objects.v1i.yolov8", "valid", "images"),
                os.path.join(BASE, "Indian traffic objects.v1i.yolov8", "valid", "labels"),
            ),
            "test": (
                os.path.join(BASE, "Indian traffic objects.v1i.yolov8", "test", "images"),
                os.path.join(BASE, "Indian traffic objects.v1i.yolov8", "test", "labels"),
            ),
        }
    },
    {
        "name": "archive_traffic",
        "yaml": os.path.join(BASE, "archive", "Traffic Dataset", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "archive", "Traffic Dataset", "images", "train"),
                os.path.join(BASE, "archive", "Traffic Dataset", "labels", "train"),
            ),
            "val": (
                os.path.join(BASE, "archive", "Traffic Dataset", "images", "val"),
                os.path.join(BASE, "archive", "Traffic Dataset", "labels", "val"),
            ),
            "test": (
                os.path.join(BASE, "archive", "Traffic Dataset", "images", "test"),
                os.path.join(BASE, "archive", "Traffic Dataset", "labels", "train"),
            ),
        }
    },
    {
        "name": "indian_vehicle_class",
        "yaml": os.path.join(BASE, "Indian Vehicle Classification.v2i.yolov8", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "Indian Vehicle Classification.v2i.yolov8", "train", "images"),
                os.path.join(BASE, "Indian Vehicle Classification.v2i.yolov8", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "Indian Vehicle Classification.v2i.yolov8", "valid", "images"),
                os.path.join(BASE, "Indian Vehicle Classification.v2i.yolov8", "valid", "labels"),
            ),
        }
    },
    {
        "name": "indian_vehicles_det",
        "yaml": os.path.join(BASE, "Indian-Vehicles-Detection.v4i.yolov8", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "Indian-Vehicles-Detection.v4i.yolov8", "train", "images"),
                os.path.join(BASE, "Indian-Vehicles-Detection.v4i.yolov8", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "Indian-Vehicles-Detection.v4i.yolov8", "valid", "images"),
                os.path.join(BASE, "Indian-Vehicles-Detection.v4i.yolov8", "valid", "labels"),
            ),
        }
    },
    {
        "name": "archive1",
        "yaml": os.path.join(BASE, "archive (1)", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "archive (1)", "train", "images"),
                os.path.join(BASE, "archive (1)", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "archive (1)", "valid", "images"),
                os.path.join(BASE, "archive (1)", "valid", "labels"),
            ),
        }
    },
    {
        "name": "archive2",
        "yaml": os.path.join(BASE, "archive (2)", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "archive (2)", "train", "images"),
                os.path.join(BASE, "archive (2)", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "archive (2)", "valid", "images"),
                os.path.join(BASE, "archive (2)", "valid", "labels"),
            ),
        }
    },
    {
        "name": "archive3",
        "yaml": os.path.join(BASE, "archive (3)", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "archive (3)", "train", "images"),
                os.path.join(BASE, "archive (3)", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "archive (3)", "valid", "images"),
                os.path.join(BASE, "archive (3)", "valid", "labels"),
            ),
        }
    },
    {
        "name": "final_dataset",
        "yaml": os.path.join(BASE, "final_dataset-20260419T192218Z-3-002", "data.yaml"),
        "splits": {
            "train": (
                os.path.join(BASE, "final_dataset-20260419T192218Z-3-002", "train", "images"),
                os.path.join(BASE, "final_dataset-20260419T192218Z-3-002", "train", "labels"),
            ),
            "val": (
                os.path.join(BASE, "final_dataset-20260419T192218Z-3-002", "valid", "images"),
                os.path.join(BASE, "final_dataset-20260419T192218Z-3-002", "valid", "labels"),
            ),
        }
    },
]

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def read_yaml_classes(yaml_path):
    if not os.path.exists(yaml_path):
        folder = os.path.dirname(yaml_path)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith(".yaml") or f.endswith(".yml"):
                    yaml_path = os.path.join(folder, f)
                    break
    if not os.path.exists(yaml_path):
        return []
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        return list(names)
    except Exception as e:
        print(f"     ⚠ yaml error: {e}")
        return []


def remap_label_file(src, dst, class_names):
    try:
        with open(src, "r") as f:
            lines = f.readlines()
    except Exception:
        open(dst, "w").close()
        return
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        old_id = int(parts[0])
        name   = class_names[old_id].lower().strip() if old_id < len(class_names) else "other"
        new_id = REMAP.get(name, 9)
        parts[0] = str(new_id)
        new_lines.append(" ".join(parts) + "\n")
    with open(dst, "w") as f:
        f.writelines(new_lines)


def copy_split(ds_name, img_dir, lbl_dir, out_split, class_names, stats):
    if not os.path.isdir(img_dir):
        return 0
    out_img = os.path.join(OUT, "images", out_split)
    out_lbl = os.path.join(OUT, "labels", out_split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    img_exts = (".jpg", ".jpeg", ".png", ".bmp",
                ".JPG", ".JPEG", ".PNG", ".BMP")
    prefix = ds_name[:6].replace(" ", "_")
    count  = 0
    for img_file in os.listdir(img_dir):
        if not img_file.endswith(img_exts):
            continue
        stem    = os.path.splitext(img_file)[0]
        ext     = os.path.splitext(img_file)[1]
        new_img = f"{prefix}_{stem}{ext}"
        new_lbl = f"{prefix}_{stem}.txt"
        shutil.copy2(os.path.join(img_dir, img_file),
                     os.path.join(out_img, new_img))
        src_lbl = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(src_lbl):
            remap_label_file(src_lbl,
                             os.path.join(out_lbl, new_lbl),
                             class_names)
        else:
            open(os.path.join(out_lbl, new_lbl), "w").close()
        count += 1
    stats[ds_name][out_split] += count
    return count


# ─────────────────────────────────────────────────────────────
#  STEP 1 — FOLDER STRUCTURE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — Creating folder structure")
print("="*60)
if os.path.exists(OUT):
    print("  ⚠ Old merged_dataset found — removing ...")
    shutil.rmtree(OUT)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
print(f"  ✅ Created: {OUT}")

# ─────────────────────────────────────────────────────────────
#  STEP 2 — MERGE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — Merging all datasets")
print("="*60)
stats = defaultdict(lambda: defaultdict(int))
total = 0
for ds in SOURCES:
    name        = ds["name"]
    class_names = read_yaml_classes(ds["yaml"])
    print(f"\n  📁 {name}")
    if class_names:
        print(f"     Classes : {class_names}")
    else:
        print(f"     ⚠ No yaml found — labels will be mapped as 'other'")
    for split_name, (img_dir, lbl_dir) in ds["splits"].items():
        n = copy_split(name, img_dir, lbl_dir, split_name, class_names, stats)
        if n > 0:
            print(f"     {split_name:5s} → {n:5d} images")
        total += n
print(f"\n  ✅ TOTAL images merged : {total}")

# ─────────────────────────────────────────────────────────────
#  STEP 3 — data.yaml
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Writing data.yaml")
print("="*60)
yaml_out = os.path.join(OUT, "data.yaml")
with open(yaml_out, "w") as f:
    yaml.dump({
        "path":  OUT,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(FINAL_CLASSES),
        "names": FINAL_CLASSES,
    }, f, default_flow_style=False, sort_keys=False)
print(f"  ✅ Saved : {yaml_out}")

# ─────────────────────────────────────────────────────────────
#  STEP 4 — VERIFY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — Verification")
print("="*60)
for split in ["train", "val", "test"]:
    n = len([f for f in os.listdir(os.path.join(OUT, "images", split))
             if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
    print(f"  {split:5s} : {n:6d} images")

print("\n  Class distribution in TRAIN labels:")
class_counts = defaultdict(int)
lbl_train = os.path.join(OUT, "labels", "train")
for lf in os.listdir(lbl_train):
    if not lf.endswith(".txt"):
        continue
    with open(os.path.join(lbl_train, lf)) as f:
        for line in f:
            p = line.strip().split()
            if p:
                class_counts[int(p[0])] += 1
for idx, cname in enumerate(FINAL_CLASSES):
    cnt    = class_counts.get(idx, 0)
    bar    = "█" * min(40, cnt // max(1, total // 500))
    status = "✅" if cnt > 0 else "⚠  NO DATA"
    print(f"  {idx}: {cname:<12} {cnt:7d} boxes  {bar} {status}")

# ─────────────────────────────────────────────────────────────
#  STEP 5 — TRAIN
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 — YOLOv8 Training")
print("="*60)

gpu = torch.cuda.is_available()
print(f"\n  GPU available : {gpu}")
if gpu:
    print(f"  GPU name      : {torch.cuda.get_device_name(0)}")
    BATCH = 16
    IMGSZ = 640
    MODEL = os.path.join(BASE, "yolov8m.pt")
else:
    print("  ⚠ No GPU — using CPU (slow)")
    BATCH = 4
    IMGSZ = 416
    MODEL = os.path.join(BASE, "yolov8s.pt")

print(f"  Model  : {os.path.basename(MODEL)}")
print(f"  Batch  : {BATCH}")
print(f"  ImgSz  : {IMGSZ}")
print(f"  Epochs : 50\n")

from ultralytics import YOLO
model = YOLO(MODEL)
model.train(
    data        = yaml_out,
    epochs      = 50,
    imgsz       = IMGSZ,
    batch       = BATCH,
    project     = os.path.join(BASE, "runs", "train"),
    name        = "traffic_v1",
    augment     = True,
    mosaic      = 1.0,
    mixup       = 0.1,
    copy_paste  = 0.1,
    degrees     = 5.0,
    fliplr      = 0.5,
    flipud      = 0.0,
    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    scale       = 0.5,
    translate   = 0.1,
    optimizer   = "AdamW",
    lr0         = 0.001,
    lrf         = 0.01,
    patience    = 15,
    workers     = 4,
    save        = True,
    save_period = 10,
    plots       = True,
    verbose     = True,
)

# ─────────────────────────────────────────────────────────────
#  STEP 6 — VALIDATE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 — Final Validation")
print("="*60)
best = os.path.join(BASE, "runs", "train", "traffic_v1", "weights", "best.pt")
metrics = YOLO(best).val(data=yaml_out)
print(f"\n  mAP@50    : {metrics.box.map50:.4f}")
print(f"  mAP@50-95 : {metrics.box.map:.4f}")
print(f"  Precision : {metrics.box.mp:.4f}")
print(f"  Recall    : {metrics.box.mr:.4f}")
print("\n" + "="*60)
print("  ✅ TRAINING COMPLETE!")
print(f"  Best model saved at:")
print(f"  {best}")
print("="*60)