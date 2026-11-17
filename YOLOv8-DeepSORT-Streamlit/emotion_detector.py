# emotion_detector.py (OpenCV + ONNX - No TensorFlow, No FER)

import cv2
import onnxruntime as ort
import numpy as np

class EmotionDetector:
    def __init__(self):
        model_path = "emotion_ferplus.onnx"
        self.session = ort.InferenceSession(model_path)

        self.emotions = [
            "neutral", "happiness", "surprise",
            "sadness", "anger", "disgust", "fear", "contempt"
        ]

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, 1))
        return face

    def detect_emotion(self, frame):
        try:
            face = self.preprocess(frame)
            inputs = {self.session.get_inputs()[0].name: face}
            outputs = self.session.run(None, inputs)[0]

            emotion_id = int(np.argmax(outputs))
            return self.emotions[emotion_id]
        except:
            return "neutral"
