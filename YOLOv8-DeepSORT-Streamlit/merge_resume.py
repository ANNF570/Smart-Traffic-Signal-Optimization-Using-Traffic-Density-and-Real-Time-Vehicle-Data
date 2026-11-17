import os, shutil, glob

BASE     = r"F:\major_project\YOLOv8-DeepSORT-Streamlit"
SRC_DIR  = os.path.join(BASE, "extra_ambulance_repo", "Src", "Dataset", "Main Dataset")
DST_IMG  = os.path.join(BASE, "yolo_dataset", "images", "train")
DST_LBL  = os.path.join(BASE, "yolo_dataset", "labels", "train")

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_LBL, exist_ok=True)

images = glob.glob(os.path.join(SRC_DIR, "*.jpeg")) + \
         glob.glob(os.path.join(SRC_DIR, "*.jpg")) + \
         glob.glob(os.path.join(SRC_DIR, "*.png"))

print(f"Found {len(images)} ambulance images")

copied = 0
for img_path in images:
    fname = f"amb_{copied:05d}.jpg"
    shutil.copy(img_path, os.path.join(DST_IMG, fname))
    lbl_name = f"amb_{copied:05d}.txt"
    with open(os.path.join(DST_LBL, lbl_name), "w") as f:
        f.write("0 0.5 0.5 0.9 0.9\n")
    copied += 1
    if copied % 500 == 0:
        print(f"  Copied {copied}/{len(images)} images...")

print(f"\n✅ Added {copied} ambulance images!")
print(f"Total train images now: {len(os.listdir(DST_IMG))}")