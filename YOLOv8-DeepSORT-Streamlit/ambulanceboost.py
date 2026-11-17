import shutil
import os

src_img = "dataset_merged/train/images"
src_lbl = "dataset_merged/train/labels"

files = os.listdir(src_img)

for file in files:
    src_img_path = os.path.join(src_img, file)

    # ✅ Skip folders
    if not os.path.isfile(src_img_path):
        continue

    # ✅ Only process ambulance images
    if "ambulance" in file.lower():
        for i in range(3):  # duplicate 3 times
            
            new_img_name = f"copy{i}_{file}"
            dst_img_path = os.path.join(src_img, new_img_name)

            shutil.copy(src_img_path, dst_img_path)

            # ✅ Handle label safely
            label_file = os.path.splitext(file)[0] + ".txt"
            src_lbl_path = os.path.join(src_lbl, label_file)

            if os.path.exists(src_lbl_path):
                new_lbl_name = f"copy{i}_{label_file}"
                dst_lbl_path = os.path.join(src_lbl, new_lbl_name)

                shutil.copy(src_lbl_path, dst_lbl_path)

print("✅ Ambulance boosted")