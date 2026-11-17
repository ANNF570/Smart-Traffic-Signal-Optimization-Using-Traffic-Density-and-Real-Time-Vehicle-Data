import base64

# NOTE: Paste PART 1, PART 2, PART 3 inside this triple-quoted string:
data = """
PASTE_PARTS_HERE
"""

# Decode Base64 → create ONNX file
onnx_bytes = base64.b64decode(data)

with open("emotion_ferplus.onnx", "wb") as f:
    f.write(onnx_bytes)

print("ONNX model created successfully: emotion_ferplus.onnx")
