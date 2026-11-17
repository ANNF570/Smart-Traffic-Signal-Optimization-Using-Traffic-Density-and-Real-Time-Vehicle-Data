import shutil
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_ROOT = "./Dataset"
OUTPUT_ROOT  = "./yolo_dataset"
# ──────────────────────────────────────────────────────────────────────────────

src_img_dir = Path(DATASET_ROOT) / "valid" / "img"
dst_img_dir = Path(OUTPUT_ROOT) / "images" / "valid"
dst_img_dir.mkdir(parents=True, exist_ok=True)

copied = 0
skipped = 0

for img_file in src_img_dir.iterdir():
    if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        dest = dst_img_dir / img_file.name
        if not dest.exists():
            shutil.copy(img_file, dest)
            copied += 1
        else:
            skipped += 1

print(f"[DONE] Copied {copied} images to {dst_img_dir}")
print(f"       Skipped {skipped} already existing")
print(f"\nNow re-run: python train.py")