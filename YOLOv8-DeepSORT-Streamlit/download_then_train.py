import os, urllib.request, time

BASE    = r"F:\major_project\YOLOv8-DeepSORT-Streamlit"
YOLO_DS = os.path.join(BASE, "yolo_dataset")
IMG_DIR = os.path.join(YOLO_DS, "images", "train")
LBL_DIR = os.path.join(YOLO_DS, "labels", "train")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# ── Ambulance image URLs (free, no login) ─────────────
URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Ambulance_Mercedes_Benz_Sprinter.jpg/640px-Ambulance_Mercedes_Benz_Sprinter.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/DRK_RTW_Mercedes_Sprinter.jpg/640px-DRK_RTW_Mercedes_Sprinter.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Delhi_Ambulance.jpg/640px-Delhi_Ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Indian_Ambulance.jpg/640px-Indian_Ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Mumbai_ambulance.jpg/640px-Mumbai_ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/SMURD_ambulance_Dacia_Duster.jpg/640px-SMURD_ambulance_Dacia_Duster.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Ambulance_at_Jipmer.jpg/640px-Ambulance_at_Jipmer.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Kerala_108_Ambulance.jpg/640px-Kerala_108_Ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Ambulance_India.jpg/640px-Ambulance_India.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Pakistani_Ambulance.jpg/640px-Pakistani_Ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Ford_Transit_ambulance.jpg/640px-Ford_Transit_ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ambulance_Bucharest.jpg/640px-Ambulance_Bucharest.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Turkish_ambulance.jpg/640px-Turkish_ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Japan_ambulance_Toyota_Himedic.jpg/640px-Japan_ambulance_Toyota_Himedic.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Chinese_ambulance.jpg/640px-Chinese_ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/thirty/BrazilianAmbulance.jpg/640px-BrazilianAmbulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/South_African_Ambulance.jpg/640px-South_African_Ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/NYC_Ambulance.jpg/640px-NYC_Ambulance.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/London_Ambulance_Service_Mercedes_Sprinter.jpg/640px-London_Ambulance_Service_Mercedes_Sprinter.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Australian_Ambulance.jpg/640px-Australian_Ambulance.jpg",
]

# ── Download ──────────────────────────────────────────
print("=" * 50)
print("STEP 1: Downloading ambulance images...")
print("=" * 50)

downloaded = 0
skipped = 0
for i, url in enumerate(URLS):
    try:
        fname = f"extra_ambulance_{i:04d}.jpg"
        fpath = os.path.join(IMG_DIR, fname)
        if os.path.exists(fpath):
            print(f"  [{i+1}/{len(URLS)}] Already exists, skipping")
            downloaded += 1
            continue
        urllib.request.urlretrieve(url, fpath)
        # Write YOLO label: class 0 (ambulance), full centered box
        lbl_path = os.path.join(LBL_DIR, fname.replace(".jpg", ".txt"))
        with open(lbl_path, "w") as f:
            f.write("0 0.5 0.5 0.9 0.9\n")
        downloaded += 1
        print(f"  [{i+1}/{len(URLS)}] Downloaded ✅ {fname}")
        time.sleep(0.2)
    except Exception as e:
        skipped += 1
        print(f"  [{i+1}/{len(URLS)}] Skipped ❌ ({e})")

print(f"\nDownloaded: {downloaded} | Skipped: {skipped}")

# ── Count total dataset ───────────────────────────────
total = len([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg','.png'))])
print(f"Total training images now: {total}")

# ── Train ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Starting YOLOv8 Training...")
print("=" * 50 + "\n")

yaml = os.path.join(YOLO_DS, "data.yaml")
os.system(f'yolo train model=yolov8n.pt data="{yaml}" epochs=100 imgsz=640 project=runs/detect/runs/detect name=my_model exist_ok=True')