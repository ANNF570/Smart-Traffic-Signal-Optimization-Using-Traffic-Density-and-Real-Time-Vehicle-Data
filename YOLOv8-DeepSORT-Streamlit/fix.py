import subprocess
import os
from pathlib import Path

dataset_dir = Path("./Siren-Sound-Dataset")

for wav_file in dataset_dir.rglob("*.wav"):
    tmp = wav_file.with_suffix(".tmp.wav")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_file),
        "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
        str(tmp)
    ], capture_output=True)
    
    if result.returncode == 0:
        os.replace(tmp, wav_file)
        print(f"✓ Converted: {wav_file.name}")
    else:
        print(f"✗ Failed: {wav_file.name} — {result.stderr.decode()[-200:]}")