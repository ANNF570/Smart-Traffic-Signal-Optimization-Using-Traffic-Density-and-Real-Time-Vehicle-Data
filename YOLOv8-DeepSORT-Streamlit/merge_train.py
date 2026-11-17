import os, shutil, zipfile

BASE    = r"F:\major_project\YOLOv8-DeepSORT-Streamlit"
ZIP     = os.path.join(BASE, "ambulance_extra.zip")
EXTRA   = os.path.join(BASE, "extra_ambulance")
YOLO_DS = os.path.join(BASE, "yolo_dataset")

# ── STEP 1: Extract ───────────────────────────────────
print("Extracting zip...")
with zipfile.ZipFile(ZIP, 'r') as z:
    z.extractall(EXTRA)
print("Extracted ✅")

# ── Show what was extracted ───────────────────────────
for root, dirs, files in os.walk(EXTRA):
    level = root.replace(EXTRA, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        for f in files[:3]:
            print(f"{indent}  {f}")