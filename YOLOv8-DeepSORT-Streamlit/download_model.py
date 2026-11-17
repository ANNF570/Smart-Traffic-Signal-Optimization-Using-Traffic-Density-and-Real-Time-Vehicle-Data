import urllib.request

url = "https://raw.githubusercontent.com/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

print("Downloading model...")

urllib.request.urlretrieve(url, "emotion_ferplus.onnx")

print("Download complete.")
