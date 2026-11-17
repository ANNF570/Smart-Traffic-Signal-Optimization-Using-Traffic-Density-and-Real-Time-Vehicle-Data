import numpy as np
import sounddevice as sd
import librosa
import threading, queue, time


class AmbulanceDetector:
    def __init__(self, yolo_model=None, sample_rate=16000, threshold=100, mic_index=1):
        """
        Detects ambulances using both YOLOv8 (visual) and siren sound (audio).
        :param yolo_model: Trained YOLOv8 model for ambulance detection
        :param sample_rate: Audio capture rate
        :param threshold: Energy threshold for siren detection
        :param mic_index: Input device index for microphone (check using sd.query_devices())
        """
        self.model = yolo_model
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.siren_detected = False
        self.q = queue.Queue()

        # 🎙️ Set correct microphone device manually
        try:
            sd.default.device = [mic_index, None]  # Input = mic_index, Output = None
            print(f"🎧 Using microphone device index: {mic_index}")
        except Exception as e:
            print("⚠️ Could not set default microphone:", e)

        # Start audio listener thread
        threading.Thread(target=self._audio_worker, daemon=True).start()
        try:
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate
            )
            self.stream.start()
            print("🎙️ Microphone listening started successfully.")
        except Exception as e:
            print("⚠️ Microphone not started:", e)

    # 🟢 Audio input callback
    def _audio_callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    # 🔄 Background audio analyzer
    def _audio_worker(self):
        while True:
            try:
                data = self.q.get()
                y = np.mean(data, axis=1)
                S = np.abs(librosa.stft(y, n_fft=1024))
                freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=1024)
                band_idx = (freqs > 500) & (freqs < 2000)
                band_energy = S[band_idx, :].mean()

                # Siren detection logic
                self.siren_detected = band_energy > self.threshold

                # 🧠 Debug info (helps tuning threshold)
                print(f"🔊 Band Energy: {band_energy:.2f} | Detected: {self.siren_detected}")

                time.sleep(0.3)  # small pause to avoid CPU overload
            except Exception as e:
                print("Audio processing error:", e)

    # 🚑 Detect ambulance visually using YOLOv8 results
    def detect_ambulance_in_frame(self, results):
        if not hasattr(results[0], "boxes"):
            return False
        for box in results[0].boxes:
            cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            cls_name = self.model.names.get(cls_id, "unknown")
            if "ambulance" in cls_name.lower():
                print(f"🚑 Visual ambulance detected: {cls_name}")
                return True
        return False

    # 🧩 Combine both modalities (visual + audio)
    def is_emergency_detected(self, results):
        visual = self.detect_ambulance_in_frame(results)
        audio = self.siren_detected
        if visual or audio:
            print(f"⚠️ Emergency detected | Visual: {visual} | Audio: {audio}")
        return visual or audio
