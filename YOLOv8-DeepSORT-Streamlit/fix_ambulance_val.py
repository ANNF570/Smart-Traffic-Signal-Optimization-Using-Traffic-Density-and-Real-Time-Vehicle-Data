import os, shutil, random

TRAIN_IMG = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\yolo_dataset\images\train"
TRAIN_LBL = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\yolo_dataset\labels\train"
VAL_IMG   = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\yolo_dataset\images\valid"
VAL_LBL   = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\yolo_dataset\labels\valid"

# Find all label files that contain ambulance (class 0)
ambulance_files = []
for f in os.listdir(TRAIN_LBL):
    if not f.endswith(".txt"): continue
    with open(os.path.join(TRAIN_LBL, f)) as r:
        if any(line.startswith("0 ") for line in r):
            ambulance_files.append(f)

print(f"Found {len(ambulance_files)} ambulance label files in train")

# Move 20 to validation
random.shuffle(ambulance_files)
to_move = ambulance_files[:20]

for lbl_file in to_move:
    stem = lbl_file.replace(".txt", "")
    # Move label
    shutil.move(os.path.join(TRAIN_LBL, lbl_file), os.path.join(VAL_LBL, lbl_file))
    # Move image
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
        img_src = os.path.join(TRAIN_IMG, stem + ext)
        if os.path.exists(img_src):
            shutil.move(img_src, os.path.join(VAL_IMG, stem + ext))
            break
    print(f"  Moved {lbl_file} ✅")

print(f"\n✅ Moved {len(to_move)} ambulance images to validation!")
print("Now retrain with:")
print('yolo train model=yolov8n.pt data="F:\\major_project\\YOLOv8-DeepSORT-Streamlit\\yolo_dataset\\data.yaml" epochs=100 imgsz=640 project=runs/detect/runs/detect name=my_model2 exist_ok=True')