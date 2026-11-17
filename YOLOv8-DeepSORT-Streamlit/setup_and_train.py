import os
import shutil
from pathlib import Path

# =========================
# 1. CREATE FOLDERS
# =========================
base = Path("dataset")
(base / "images/train").mkdir(parents=True, exist_ok=True)
(base / "images/val").mkdir(parents=True, exist_ok=True)
(base / "labels/train").mkdir(parents=True, exist_ok=True)
(base / "labels/val").mkdir(parents=True, exist_ok=True)

print("Folders created ✅")

# =========================
# 2. DOWNLOAD COCO128 (FAST)
# =========================
import urllib.request
import zipfile

url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
zip_path = "coco128.zip"

if not os.path.exists(zip_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")

print("Dataset downloaded ✅")

# =========================
# 3. COPY IMAGES + LABELS
# =========================
src_images = Path("coco128/images/train2017")
src_labels = Path("coco128/labels/train2017")

images = list(src_images.glob("*.jpg"))

for i, img in enumerate(images):
    if i < int(len(images)*0.8):
        shutil.copy(img, base / "images/train" / img.name)
        label = src_labels / (img.stem + ".txt")
        if label.exists():
            shutil.copy(label, base / "labels/train" / label.name)
    else:
        shutil.copy(img, base / "images/val" / img.name)
        label = src_labels / (img.stem + ".txt")
        if label.exists():
            shutil.copy(label, base / "labels/val" / label.name)

print("Data copied ✅")

# =========================
# 4. CREATE dataset.yaml
# =========================
yaml_content = """path: dataset
train: images/train
val: images/val

names:
  0: car
  1: bike
  2: bus
  3: truck
  4: auto
  5: ambulance
  6: person
  7: traffic light
  8: police
"""

with open("dataset.yaml", "w") as f:
    f.write(yaml_content)

print("dataset.yaml created ✅")

# =========================
# 5. START TRAINING (FAST)
# =========================
print("Starting training... 🚀")

os.system(
    "yolo detect train model=yolov8n.pt data=dataset.yaml "
    "epochs=10 imgsz=416 batch=4 workers=0 device=cpu fraction=0.3"
)