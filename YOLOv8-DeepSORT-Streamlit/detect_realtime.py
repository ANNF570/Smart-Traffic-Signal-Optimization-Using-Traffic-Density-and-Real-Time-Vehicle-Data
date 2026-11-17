"""
Real-Time Ambulance Siren Detector — PyTorch Version
Records mic audio → generates spectrogram PNG → classifies with CNN.
Run after train_siren.py
"""

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import pickle
import time
import warnings
import io
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")           # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_SAVE   = "./siren_model.pt"
ENCODER_SAVE = "./label_encoder.pkl"
SAMPLE_RATE  = 22050
DURATION     = 3
IMG_SIZE     = (64, 64)
THRESHOLD    = 0.75
# ──────────────────────────────────────────────────────────────────────────────

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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.classifier(self.features(x))


def audio_to_spectrogram_tensor(audio: np.ndarray) -> torch.Tensor:
    """
    Convert a raw audio array to a (1, 3, H, W) tensor by:
    1. Computing a mel spectrogram
    2. Rendering it to a PNG in memory (matches training format)
    3. Resizing to IMG_SIZE and normalising to [0, 1]
    """
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Render to PNG in memory (no file I/O)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=32)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    librosa.display.specshow(S_db, sr=SAMPLE_RATE, ax=ax)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3)
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
model = SirenCNN()
model.load_state_dict(torch.load(MODEL_SAVE, map_location="cpu"))
model.eval()

with open(ENCODER_SAVE, "rb") as f:
    le = pickle.load(f)

siren_idx = list(le.classes_).index("siren")

print("[INFO] Model loaded. Starting real-time detection...\n")
print("=" * 55)
print("  🚑  AMBULANCE SIREN DETECTOR — LIVE")
print(f"  Chunk: {DURATION}s | Alert threshold: {THRESHOLD*100:.0f}%")
print("  Press Ctrl+C to stop")
print("=" * 55 + "\n")

# ─── DETECTION LOOP ───────────────────────────────────────────────────────────
try:
    while True:
        # Record audio
        audio = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Convert to spectrogram tensor and predict
        tensor = audio_to_spectrogram_tensor(audio)
        with torch.no_grad():
            raw_prob = model(tensor).item()

        # raw_prob is P(class with index 1); adjust if siren_idx != 1
        confidence = raw_prob if siren_idx == 1 else 1 - raw_prob

        bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        ts  = time.strftime("%H:%M:%S")

        if confidence >= THRESHOLD:
            print(f"[{ts}] 🚨 AMBULANCE SIREN DETECTED! [{bar}] {confidence*100:.1f}%")
        else:
            print(f"[{ts}] ✅ No siren              [{bar}] {confidence*100:.1f}%")

except KeyboardInterrupt:
    print("\n[INFO] Detector stopped.")