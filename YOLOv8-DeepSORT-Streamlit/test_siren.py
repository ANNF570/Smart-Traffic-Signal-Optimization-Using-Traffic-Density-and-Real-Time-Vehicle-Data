import numpy as np
import librosa
from ambulance_detector import AmbulanceDetector

# Path to your siren file
siren_path = "sounds/ambulance_siren.wav"

# Initialize detector (no YOLO model needed)
detector = AmbulanceDetector(yolo_model=None)

# Load siren audio file
print(f"🎧 Loading test siren from: {siren_path}")
y, sr = librosa.load(siren_path, sr=detector.sample_rate)

# Perform Short-Time Fourier Transform (STFT)
S = np.abs(librosa.stft(y, n_fft=1024))
freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

# Focus on siren frequency band (500–2000 Hz)
band_idx = (freqs > 500) & (freqs < 2000)
band_energy = S[band_idx, :].mean()

# Check if above threshold
detected = band_energy > detector.threshold

print(f"\n🔊 Band Energy: {band_energy:.2f}")
print(f"🚨 Siren Detected: {detected}")

if detected:
    print("✅ SUCCESS: The ambulance siren was detected correctly!")
else:
    print("❌ No siren detected — adjust threshold or use a clearer siren sound.")
