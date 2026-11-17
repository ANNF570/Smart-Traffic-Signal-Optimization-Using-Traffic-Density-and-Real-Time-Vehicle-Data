"""
Supervisely Format → YOLO Format Converter
============================================
Converts your dataset structure:
    Dataset/
    ├── train/
    │   ├── ann/   ← Supervisely JSON annotations
    │   └── img/   ← images
    ├── valid/
    │   ├── ann/
    │   └── img/
    └── meta.json

To YOLO structure:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

Usage:
    python supervisely_to_yolo.py --dataset ./Dataset --output ./yolo_dataset
"""

import os
import json
import shutil
import argparse
from pathlib import Path


def load_meta(dataset_path):
    """Load class names from meta.json"""
    meta_path = os.path.join(dataset_path, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Extract class names in order
    classes = []
    for cls in meta.get("classes", []):
        classes.append(cls["title"])

    print(f"\n✅ Found {len(classes)} classes in meta.json:")
    for i, c in enumerate(classes):
        print(f"   {i}: {c}")
    return classes


def convert_annotation(ann_path, class_names, img_w, img_h):
    """
    Convert a single Supervisely JSON annotation to YOLO format lines.
    Returns list of strings like: "class_id cx cy w h"
    """
    with open(ann_path, "r") as f:
        ann = json.load(f)

    yolo_lines = []

    for obj in ann.get("objects", []):
        class_title = obj.get("classTitle", "")
        if class_title not in class_names:
            print(f"   ⚠️  Skipping unknown class: {class_title}")
            continue

        class_id = class_names.index(class_title)
        geo_type = obj.get("geometryType", "")

        # Handle bounding box (rectangle)
        if geo_type == "rectangle":
            points = obj["labelsMap"] if "labelsMap" in obj else None
            exterior = obj.get("points", {}).get("exterior", [])

            if exterior and len(exterior) == 2:
                (x1, y1), (x2, y2) = exterior[0], exterior[1]
            elif "labelsMap" not in obj:
                # Try alternative structure
                try:
                    exterior = obj["points"]["exterior"]
                    (x1, y1), (x2, y2) = exterior[0], exterior[1]
                except Exception:
                    continue
            else:
                continue

            # Convert to YOLO format (normalized cx, cy, w, h)
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = abs(x2 - x1) / img_w
            h  = abs(y2 - y1) / img_h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w  = max(0, min(1, w))
            h  = max(0, min(1, h))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Handle polygon → convert to bounding box
        elif geo_type == "polygon":
            exterior = obj.get("points", {}).get("exterior", [])
            if not exterior:
                continue
            xs = [p[0] for p in exterior]
            ys = [p[1] for p in exterior]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = abs(x2 - x1) / img_w
            h  = abs(y2 - y1) / img_h

            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w  = max(0, min(1, w))
            h  = max(0, min(1, h))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        else:
            print(f"   ⚠️  Unsupported geometry type: {geo_type}, skipping.")

    return yolo_lines


def process_split(split_name, dataset_path, output_path, class_names):
    """Process train or valid split"""
    yolo_split = "val" if split_name == "valid" else split_name

    ann_dir = os.path.join(dataset_path, split_name, "ann")
    img_dir = os.path.join(dataset_path, split_name, "img")

    out_img_dir   = os.path.join(output_path, "images", yolo_split)
    out_label_dir = os.path.join(output_path, "labels", yolo_split)

    os.makedirs(out_img_dir,   exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    ann_files = list(Path(ann_dir).glob("*.json"))
    print(f"\n📂 Processing '{split_name}' → {len(ann_files)} annotation files...")

    converted = 0
    skipped   = 0

    for ann_file in ann_files:
        # Load annotation
        with open(ann_file, "r") as f:
            ann_data = json.load(f)

        img_h = ann_data.get("size", {}).get("height", 0)
        img_w = ann_data.get("size", {}).get("width",  0)

        if img_h == 0 or img_w == 0:
            print(f"   ⚠️  Zero size for {ann_file.name}, skipping.")
            skipped += 1
            continue

        # Find corresponding image file
        base_name = ann_file.stem  # e.g. "image1.jpg" (ann files are named like "image1.jpg.json")
        # Supervisely names ann files as <image_name>.json
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        img_file = None

        for ext in img_extensions:
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_file = candidate
                break

        # Also try: ann file might be named "img.jpg.json" → image is "img.jpg"
        if img_file is None:
            for ext in img_extensions:
                if base_name.endswith(ext):
                    candidate = os.path.join(img_dir, base_name)
                    if os.path.exists(candidate):
                        img_file = candidate
                        break

        if img_file is None:
            print(f"   ⚠️  No image found for {ann_file.name}, skipping.")
            skipped += 1
            continue

        # Convert annotation
        yolo_lines = convert_annotation(str(ann_file), class_names, img_w, img_h)

        # Copy image
        img_dest = os.path.join(out_img_dir, os.path.basename(img_file))
        shutil.copy2(img_file, img_dest)

        # Write label file (even if empty — YOLO needs it)
        label_name = Path(os.path.basename(img_file)).stem + ".txt"
        label_dest = os.path.join(out_label_dir, label_name)
        with open(label_dest, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"   ✅ Converted: {converted}  |  ⚠️  Skipped: {skipped}")
    return converted


def write_yaml(output_path, class_names):
    """Write data.yaml for YOLO training"""
    yaml_content = f"""# YOLO Dataset Configuration
# Auto-generated by supervisely_to_yolo.py

path: {os.path.abspath(output_path)}
train: images/train
val:   images/val

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""
    yaml_path = os.path.join(output_path, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n📄 data.yaml written → {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Supervisely dataset to YOLO format")
    parser.add_argument("--dataset", type=str, default="./Dataset",
                        help="Path to your Supervisely dataset folder (default: ./Dataset)")
    parser.add_argument("--output",  type=str, default="./yolo_dataset",
                        help="Output folder for YOLO dataset (default: ./yolo_dataset)")
    args = parser.parse_args()

    dataset_path = args.dataset
    output_path  = args.output

    print("=" * 55)
    print("   Supervisely → YOLO Format Converter")
    print("=" * 55)
    print(f"📁 Input  : {dataset_path}")
    print(f"📁 Output : {output_path}")

    # Validate input
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset folder not found: {dataset_path}")
        return

    if not os.path.exists(os.path.join(dataset_path, "meta.json")):
        print(f"\n❌ meta.json not found in: {dataset_path}")
        return

    os.makedirs(output_path, exist_ok=True)

    # Load classes
    class_names = load_meta(dataset_path)

    # Process splits
    total = 0
    for split in ["train", "valid"]:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            total += process_split(split, dataset_path, output_path, class_names)
        else:
            print(f"\n⚠️  Split '{split}' not found, skipping.")

    # Write data.yaml
    write_yaml(output_path, class_names)

    print("\n" + "=" * 55)
    print(f"🎉 Done! Total images converted: {total}")
    print(f"\n📂 Output structure:")
    print(f"   {output_path}/")
    print(f"   ├── images/train/   ← training images")
    print(f"   ├── images/val/     ← validation images")
    print(f"   ├── labels/train/   ← YOLO .txt labels")
    print(f"   ├── labels/val/     ← YOLO .txt labels")
    print(f"   └── data.yaml       ← use this for YOLO training")
    print(f"\n🚀 Train with:")
    print(f"   yolo detect train data={output_path}/data.yaml model=yolov8n.pt epochs=100")
    print("=" * 55)


if __name__ == "__main__":
    main()