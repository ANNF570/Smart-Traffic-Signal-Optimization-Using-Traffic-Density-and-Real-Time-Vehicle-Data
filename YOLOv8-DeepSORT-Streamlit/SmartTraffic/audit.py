"""
BONUS — Verify your unified dataset before training
Run this after step3_merge_dataset.py to catch any issues early.
"""

import os
import glob
from collections import Counter
from pathlib import Path

UNIFIED_DIR = "C:/SmartTraffic/datasets/unified"

CLASS_NAMES = [
    "ambulance", "auto", "bicycle", "bus", "car",
    "concrete_mixer", "tempo", "tractor",
    "truck_tanker", "two_wheelers", "vehicle_truck",
]

print("=" * 55)
print("  DATASET AUDIT REPORT")
print("=" * 55)

for split in ["train", "val"]:
    img_dir = f"{UNIFIED_DIR}/images/{split}"
    lbl_dir = f"{UNIFIED_DIR}/labels/{split}"

    images = glob.glob(f"{img_dir}/**/*.*", recursive=True)
    labels = glob.glob(f"{lbl_dir}/**/*.txt", recursive=True)

    # Count class distribution
    class_counts = Counter()
    empty_labels  = 0

    for lbl_path in labels:
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            empty_labels += 1
            continue
        for line in lines:
            cls_id = int(line.split()[0])
            class_counts[cls_id] += 1

    print(f"\n  [{split.upper()}]")
    print(f"    Images        : {len(images)}")
    print(f"    Labels        : {len(labels)}")
    print(f"    Empty labels  : {empty_labels}")
    print(f"\n    Class distribution:")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = class_counts.get(cls_id, 0)
        bar   = "█" * min(count // 10, 50)
        print(f"      [{cls_id:2d}] {name:20s}  {count:5d}  {bar}")

# Check for images without labels
print("\n  Checking for missing labels...")
missing = 0
for split in ["train", "val"]:
    img_dir = f"{UNIFIED_DIR}/images/{split}"
    lbl_dir = f"{UNIFIED_DIR}/labels/{split}"
    for img_path in glob.glob(f"{img_dir}/**/*.*", recursive=True):
        stem     = Path(img_path).stem
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            print(f"  [MISSING LABEL] {img_path}")
            missing += 1

if missing == 0:
    print("  ✅ All images have corresponding label files")
else:
    print(f"  ⚠️  {missing} images are missing label files")

print("\n" + "=" * 55)
print("  Audit complete. Fix any issues before training.")
print("=" * 55)