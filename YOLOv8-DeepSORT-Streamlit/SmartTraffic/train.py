"""
STEP 4 — Train YOLOv8 on the unified Smart Traffic dataset
Fixed for CPU-only training (no CUDA GPU detected)
"""

import torch
from ultralytics import YOLO

# ── DEVICE DETECTION ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = 0
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️  No GPU detected — training on CPU (will be slower)")
    print("   Tip: Use smaller model (yolov8n) and fewer epochs on CPU")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_YAML = "F:/major_project/YOLOv8-DeepSORT-Streamlit/SmartTraffic/datasets/unified/data.yaml"

MODEL    = "yolov8n.pt"
EPOCHS   = 50
IMG_SIZE = 416
BATCH    = 4
PROJECT  = "runs"          # relative path — saves next to train.py
RUN_NAME = "smart_traffic_v1"

print(f"\n📋 Training config:")
print(f"   Model   : {MODEL}")
print(f"   Device  : {device}")
print(f"   Epochs  : {EPOCHS}")
print(f"   ImgSize : {IMG_SIZE}")
print(f"   Batch   : {BATCH}")
print(f"   Output  : {PROJECT}/{RUN_NAME}\n")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
model = YOLO(MODEL)

results = model.train(
    data      = DATA_YAML,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH,
    project   = PROJECT,
    name      = RUN_NAME,
    patience  = 15,
    save      = True,
    plots     = True,
    workers   = 2,         # keep low on CPU
    device    = device,    # auto-selected above
    amp       = False,     # AMP requires CUDA, must be False on CPU
    cache     = False,
    augment   = True,
)

print("\n✅ Training complete!")
print(f"📁 Best weights : {PROJECT}/{RUN_NAME}/weights/best.pt")

# ── VALIDATION ────────────────────────────────────────────────────────────────
print("\n🔍 Validating best weights...")
best_model = YOLO(f"{PROJECT}/{RUN_NAME}/weights/best.pt")
metrics = best_model.val(data=DATA_YAML)

print(f"\n📊 mAP50      : {metrics.box.map50:.4f}")
print(f"📊 mAP50-95   : {metrics.box.map:.4f}")
print(f"📊 Precision  : {metrics.box.mp:.4f}")
print(f"📊 Recall     : {metrics.box.mr:.4f}")