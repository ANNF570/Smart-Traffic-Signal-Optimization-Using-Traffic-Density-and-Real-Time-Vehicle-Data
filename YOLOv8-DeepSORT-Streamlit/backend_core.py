# backend_core.py
import cv2
from ultralytics import YOLO
from weather import get_weather
from ambulance_detector import AmbulanceDetector
from emotion_detector import EmotionDetector
from controller import AdaptiveSignalController
from data_integration import DataIntegrator
import config

# Load model only once
model = YOLO(config.DETECTION_MODEL_DIR / config.DETECTION_MODEL_LIST[0])
controller = AdaptiveSignalController()
data_integrator = DataIntegrator()
emotion_module = EmotionDetector()
ambulance_module = AmbulanceDetector(model)

def process_frame(frame, conf=0.5):
    """Runs YOLO + DeepSORT + adaptive control + emotion + weather"""
    results = model.track(frame, conf=conf, persist=True)
    annotated = results[0].plot()

    # Lane counting
    frame_h, frame_w = frame.shape[:2]
    lane_counts = [0, 0, 0, 0]
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx = int((x1 + x2) / 2)
        lane_id = min(int(cx * 4 // frame_w), 3)
        lane_counts[lane_id] += 1

    # Emotion detection
    emotion = emotion_module.detect_emotion(frame)

    # Ambulance detection
    emergency = ambulance_module.is_emergency_detected(results)

    # Weather
    weather_condition = get_weather()
    if weather_condition in ["Rain", "Thunderstorm", "Fog"]:
        controller.adjust_timing(factor=1.5)
    else:
        controller.adjust_timing(factor=1.0)

    # Signal control
    signal_status = controller.update_signal(lane_counts, emergency)

    # Log data
    data_integrator.log_event(
        lane_counts=lane_counts,
        weather=weather_condition,
        ambulance=emergency,
        signal_lane=controller.current_lane + 1,
        signal_status=signal_status,
        driver_emotion=emotion
    )

    # Pack results
    state = {
        "lane_counts": lane_counts,
        "signal_status": signal_status,
        "ambulance_detected": emergency,
        "siren_detected": getattr(ambulance_module, "siren_detected", False),
        "weather": weather_condition,
        "emotion": emotion,
        "logs": data_integrator.data.tail(5).to_dict(orient="records"),
    }
    return annotated, state
