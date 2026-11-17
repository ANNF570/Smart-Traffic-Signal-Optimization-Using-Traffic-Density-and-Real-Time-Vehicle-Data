import os

folders = [
    "dataset_merged/train/labels",
    "dataset_merged/valid/labels"
]

for folder in folders:
    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        if not path.endswith(".txt"):
            continue

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()

            if parts[0] == "0":
                parts[0] = "4"

            new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

print("✅ LABELS FIXED SUCCESSFULLY")