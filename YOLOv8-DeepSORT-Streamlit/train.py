from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────
YAML_PATH  = "./yolo_dataset/dataset.yaml"
MODEL      = "yolov8n.pt"   # 'n' = nano — fastest/lightest, best for CPU
EPOCHS     = 50
IMG_SIZE   = 640
BATCH_SIZE = 4              # Keep low on CPU to avoid memory issues
# ──────────────────────────────────────────────────────────────────────────────

model = YOLO(MODEL)

results = model.train(
    data      = YAML_PATH,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH_SIZE,
    device    = "cpu",        # Force CPU training
    workers   = 2,            # Reduce if you get memory errors
    patience  = 20,           # Early stopping if no improvement
    save      = True,
    project   = "runs/detect",
    name      = "my_model",
    verbose   = True,
)

print("\n[DONE] Training complete!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")