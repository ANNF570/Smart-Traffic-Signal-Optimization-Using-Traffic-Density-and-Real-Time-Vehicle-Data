"""
Ambulance Siren Detection - PyTorch Training Script
Dataset: https://github.com/vishnu-u/Siren-Sound-Dataset
Trains on PNG spectrogram images (not raw audio).
"""

import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH = "./Siren-Sound-Dataset"
MODEL_SAVE   = "./siren_model.pt"
ENCODER_SAVE = "./label_encoder.pkl"
IMG_SIZE     = (64, 64)      # resize all PNGs to this
EPOCHS       = 50
BATCH_SIZE   = 16
LR           = 0.001
# ──────────────────────────────────────────────────────────────────────────────

def load_image(file_path):
    """Load a PNG spectrogram and flatten to a 1D feature vector."""
    try:
        img = Image.open(file_path).convert("RGB").resize(IMG_SIZE)
        return np.array(img, dtype=np.float32).flatten() / 255.0
    except Exception as e:
        print(f"  [WARN] Skipping {file_path}: {e}")
        return None

print("[INFO] Loading spectrogram PNG images...")
features, labels = [], []

class_map = {
    "ambulance": "siren",
    "firetruck": "siren",
    "traffic":   "no_siren"
}

for folder, label in class_map.items():
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.exists(folder_path):
        print(f"  [WARN] Folder not found: {folder_path}")
        continue
    png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    print(f"  {folder}: {len(png_files)} PNG files → label='{label}'")
    for png_file in png_files:
        feat = load_image(os.path.join(folder_path, png_file))
        if feat is not None:
            features.append(feat)
            labels.append(label)

X = np.array(features)
y = np.array(labels)
print(f"\n[INFO] Total samples: {len(X)}")
print(f"[INFO] Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

if len(X) == 0:
    raise RuntimeError("No PNG files found! Check DATASET_PATH and folder structure.")

le = LabelEncoder()
y_enc = le.fit_transform(y).astype(np.float32)
print(f"[INFO] Classes: {le.classes_}")

with open(ENCODER_SAVE, "wb") as f:
    pickle.dump(le, f)

# ─── DATASET ──────────────────────────────────────────────────────────────────
class SirenDataset(Dataset):
    def __init__(self, X, y):
        # Shape: (N, 3, H, W) for CNN
        self.X = torch.tensor(X).reshape(-1, 3, IMG_SIZE[0], IMG_SIZE[1])
        self.y = torch.tensor(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
train_loader = DataLoader(SirenDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(SirenDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

# ─── MODEL ────────────────────────────────────────────────────────────────────
class SirenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        # After 3x MaxPool2d(2): 64 -> 32 -> 16 -> 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.classifier(self.features(x))

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Using device: {device}")
model     = SirenCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("[INFO] Training started...\n")

best_val_loss    = float("inf")
patience_counter = 0
PATIENCE         = 10

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct = 0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += ((out > 0.5).float() == y_b).sum().item()

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out  = model(X_b)
            loss = criterion(out, y_b)
            val_loss    += loss.item()
            val_correct += ((out > 0.5).float() == y_b).sum().item()

    train_acc = train_correct / len(X_train) * 100
    val_acc   = val_correct   / len(X_test)  * 100
    avg_val   = val_loss / len(test_loader)

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Val Loss: {avg_val:.4f}")

    scheduler.step(avg_val)
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), MODEL_SAVE)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

# ─── EVALUATION ───────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        out = model(X_b.to(device))
        all_preds.extend((out > 0.5).cpu().numpy().flatten())
        all_true.extend(y_b.numpy().flatten())

print("\n[INFO] Classification Report:")
print(classification_report(all_true, all_preds, target_names=le.classes_))
print(f"\n[DONE] Model saved → {MODEL_SAVE}")
print("[DONE] Run next: python detect_realtime.py")