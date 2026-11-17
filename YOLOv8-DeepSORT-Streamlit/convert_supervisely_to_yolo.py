import os
import json
import shutil
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_ROOT = "./Dataset"       # Path to your Dataset folder
OUTPUT_ROOT  = "./yolo_dataset"  # Where YOLO-format data will be saved
# ──────────────────────────────────────────────────────────────────────────────

def load_meta(dataset_root):
    meta_path = os.path.join(dataset_root, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    classes = [cls["title"] for cls in meta["classes"]]
    print(f"[INFO] Found {len(classes)} classes: {classes}")
    return classes

def convert_split(dataset_root, split, classes, output_root):
    ann_dir = Path(dataset_root) / split / "ann"
    img_dir = Path(dataset_root) / split / "img"

    out_img_dir = Path(output_root) / "images" / split
    out_lbl_dir = Path(output_root) / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ann_files = list(ann_dir.glob("*.json"))
    print(f"[INFO] Converting '{split}' split — {len(ann_files)} annotations")

    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)

        img_h = ann["size"]["height"]
        img_w = ann["size"]["width"]

        # Find matching image file (any extension)
        stem = ann_file.stem  # filename without .json
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"  [WARN] No image found for {stem}, skipping.")
            continue

        # Copy image
        shutil.copy(img_file, out_img_dir / img_file.name)

        # Write YOLO label
        label_lines = []
        for obj in ann.get("objects", []):
            class_name = obj["classTitle"]
            if class_name not in classes:
                print(f"  [WARN] Unknown class '{class_name}', skipping object.")
                continue
            class_id = classes.index(class_name)

            geo_type = obj["geometryType"]
            if geo_type == "rectangle":
                pts = obj["points"]["exterior"]
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
            elif geo_type == "bbox":
                pts = obj["points"]["exterior"]
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
            else:
                # Fallback: bounding box from any polygon
                pts = obj["points"]["exterior"]
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Normalize to [0, 1]
            cx = ((x_min + x_max) / 2) / img_w
            cy = ((y_min + y_max) / 2) / img_h
            bw = (x_max - x_min) / img_w
            bh = (y_max - y_min) / img_h

            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_lbl_dir / (stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    print(f"[INFO] '{split}' split done.")

def write_yaml(output_root, classes):
    yaml_content = f"""# YOLOv8 Dataset Config
path: {os.path.abspath(output_root)}
train: images/train
val: images/valid

nc: {len(classes)}
names: {classes}
"""
    yaml_path = Path(output_root) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[INFO] Saved dataset.yaml → {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    classes = load_meta(DATASET_ROOT)
    convert_split(DATASET_ROOT, "train", classes, OUTPUT_ROOT)
    convert_split(DATASET_ROOT, "valid", classes, OUTPUT_ROOT)
    write_yaml(OUTPUT_ROOT, classes)
    print("\n[DONE] Conversion complete! Run train.py next.")