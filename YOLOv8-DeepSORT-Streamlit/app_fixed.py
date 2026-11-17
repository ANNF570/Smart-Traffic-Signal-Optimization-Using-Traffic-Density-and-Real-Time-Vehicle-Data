# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║   SMART CITY TRAFFIC CONTROL SYSTEM  v3.0                   ║
║   YOLOv8 + DeepSORT + EasyOCR + Streamlit                   ║
║   CCTV Mode | Auto Fine | Helmet | Ambulance Priority        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys, time, platform, threading, warnings, csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import plotly.express as px_plot
import streamlit as st
import streamlit.components.v1 as components

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import easyocr

warnings.filterwarnings("ignore")

if platform.system() == "Windows":
    try:
        import winsound
        WINSOUND_OK = True
    except Exception:
        WINSOUND_OK = False
else:
    WINSOUND_OK = False

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
#  All system-wide tunable parameters are defined here so they
#  can be changed in one place without hunting through the code.
# ══════════════════════════════════════════════════════════════

MODEL_NAME     = "yolov8n.pt"   # YOLOv8 nano model — fast enough for real-time
BASE_GREEN     = 5   # Reduced for faster signal transitions              # Minimum green-light duration in seconds
MAX_GREEN      = 20  # Reduced max green time              # Maximum green-light duration in seconds
YELLOW_TIME    = 2   # Reduced yellow time               # Fixed yellow-light duration in seconds
SPEED_LIMIT    = 60              # Default speed limit (km/h) — overridable in sidebar
STOP_LINE_FRAC = 0.62            # Y-fraction of frame where the stop line sits
PIXEL_TO_KMH   = 0.08            # Calibration: pixels/frame → km/h conversion factor
BUS_WEIGHT     = 3               # Buses count as 3 vehicle-units for signal timing
VEHICLE_WEIGHT = 1               # Standard vehicles count as 1 unit
LOG_FILE       = "traffic_log.csv"   # Per-frame traffic count log path
FINE_LOG       = "fine_log.csv"      # Violation fine audit log path

# Fine amounts in Indian Rupees (₹) per violation type.
# Ambulance/emergency vehicles are automatically exempted in trigger_alert().
FINE_TABLE = {
    "Red Light Violation": 1000,
    "Overspeed":           2000,
    "Helmet Violation":     500,
    "Mobile Usage":         500,
    "Wrong Side":          1500,
}

# CCTV IP Camera URLs — fill with your actual RTSP/HTTP addresses per lane.
# Leave empty strings to fall back to uploaded video files.
CCTV_URLS = ["", "", "", ""]

LOCATIONS = {
    "Tamil Nadu": {
        "Chennai": {
            "Tambaram":  ["Indira Nagar Junction", "Camp Road Junction", "Mupet Junction"],
            "Velachery": ["Velachery Main Jn", "Taramani Junction"],
        },
        "Coimbatore": {
            "RS Puram": ["Gandhipuram Junction", "Town Hall Junction"],
        },
    }
}
JUNCTION_LANES = {
    "Indira Nagar Junction": 4, "Camp Road Junction": 3, "Mupet Junction": 4,
    "Velachery Main Jn": 4,     "Taramani Junction": 3,
    "Gandhipuram Junction": 4,  "Town Hall Junction": 3,
}
LOCATION_COORDS = {
    "Chennai":    (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
}

# ══════════════════════════════════════════════════════════════
#  SESSION INIT
#  Streamlit re-runs the entire script on every interaction.
#  session_state is the only persistent store across re-runs.
#  init_session() sets default values ONCE per browser session;
#  subsequent re-runs skip keys that already exist.
# ══════════════════════════════════════════════════════════════

def init_session():
    # Default values for every key the app needs.
    # Only written if the key is not already present in session_state.
    defaults = {
        "video_started":    False,    # True once videos are loaded and detection begins
        "video_paths":      [],       # List of file paths or RTSP URLs for each lane
        "alert_active":     False,    # Whether an alert banner is currently visible
        "alert_time":       0.0,      # Unix timestamp when the alert was triggered
        "current_alert":    "",       # Human-readable alert message string
        "violations":       [],       # List of violation dicts logged this session
        "traffic_history":  [],       # Rolling list of total vehicle counts (last 60 ticks)
        "lane_counts":      [0, 0, 0, 0],  # Current per-lane vehicle count
        "bus_count":        0,        # Raw bus detections (weighted in t_score)
        "controller":       None,     # TrafficSignalController instance
        "last_alert_time":  0.0,      # Throttle: minimum 4 s between alert triggers
        "last_siren_check": 0.0,      # Throttle: siren detection runs every 8 s
        "prev_time":        time.time(),   # Used to compute FPS between frames
        "frame_skip":       0,        # Incremented each re-run; inference runs every 2nd
        "frame_count":      0,        # Absolute frame counter used to throttle OCR
        "prev_positions":   {},       # track_id → (cx, cy, timestamp) for speed calc
        "detected_plates":  [],       # Unique number plates found by EasyOCR
        "emergency_active": False,    # Sidebar emergency toggle state
        "ambulance_active": False,    # True when ambulance is detected or toggled
        "ambulance_lane":   0,        # Which lane has the active ambulance
        "siren_active":     False,    # True when audio siren is detected
        "total_fines":      0,        # Running total of all fines issued (₹)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  HELPERS
#  HiddenPrints: context manager that silences stdout.
#  Used when calling EasyOCR to suppress verbose library output
#  that would otherwise flood the Streamlit terminal log.
# ══════════════════════════════════════════════════════════════

class HiddenPrints:
    def __enter__(self):
        self._o = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, *a):
        sys.stdout.close()
        sys.stdout = self._o

# ══════════════════════════════════════════════════════════════
#  TRAFFIC SIGNAL CONTROLLER
#  ─────────────────────────────────────────────────────────────
#  Manages the traffic signal phase state machine for all lanes.
#  Logic:
#    • Normally cycles GREEN → YELLOW → GREEN for the busiest lane.
#    • Green duration scales linearly with vehicle count (up to MAX_GREEN).
#    • If ambulance/siren detected: immediately switches to the
#      ambulance lane and holds green until emergency clears.
#  Methods exposed to the rest of the app:
#    get_light(lane)      → "GREEN" | "YELLOW" | "RED"
#    time_remaining(lane) → seconds left in current phase
# ══════════════════════════════════════════════════════════════

class TrafficSignalController:
    def __init__(self, num_lanes=4):
        self.num_lanes   = num_lanes
        self.current     = 0            # index of lane currently receiving GREEN
        self.phase       = "GREEN"      # current phase: GREEN | YELLOW
        self.t_start     = time.time()  # when current phase began
        self.green_times = [BASE_GREEN] * num_lanes   # per-lane green durations

    def update(self, lane_counts, ambulance_lane=-1, siren=False):
        """Advance the signal state machine by one tick.
        Called once per Streamlit re-run after vehicle counts are updated.
        """
        now     = time.time()
        elapsed = now - self.t_start

        # ── Emergency override: hold green on ambulance lane ──
        if ambulance_lane >= 0 or siren:
            target = ambulance_lane if ambulance_lane >= 0 else self.current
            if self.current != target:
                self.current = target
                self.phase   = "GREEN"
                self.t_start = now
            return

        # ── Do nothing if no vehicles have been detected yet ──
        if not lane_counts or max(lane_counts) == 0:
            return

        # ── Recalculate green times based on current vehicle counts ──
        # More vehicles → longer green (capped at MAX_GREEN)
        for i, cnt in enumerate(lane_counts):
            self.green_times[i] = min(BASE_GREEN + cnt * 1.8, MAX_GREEN)

        cur_green = self.green_times[self.current]

        # ── Phase transitions ──
        if self.phase == "GREEN" and elapsed >= cur_green:
            self.phase   = "YELLOW"
            self.t_start = now
        elif self.phase == "YELLOW" and elapsed >= YELLOW_TIME:
            # Switch to the lane with the most vehicles
            self.phase   = "GREEN"
            self.current = lane_counts.index(max(lane_counts))
            self.t_start = now

    def get_light(self, lane):
        """Return the signal colour for a given lane index."""
        if lane == self.current:
            return self.phase
        return "RED"

    def time_remaining(self, lane):
        """Return seconds remaining in the current phase for a lane.
        Returns '—' for red lanes (time until green is non-trivial to compute).
        """
        elapsed = time.time() - self.t_start
        light   = self.get_light(lane)
        if light == "GREEN":
            return max(0, int(self.green_times[self.current] - elapsed))
        elif light == "YELLOW":
            return max(0, int(YELLOW_TIME - elapsed))
        return "—"

# ══════════════════════════════════════════════════════════════
#  ALERT SYSTEM
#  ─────────────────────────────────────────────────────────────
#  Three components:
#    _beep()         — plays an audio beep on Windows via winsound
#    send_push(msg)  — fires an ntfy.sh push notification (non-blocking)
#    trigger_alert() — central dispatcher: records violation, logs CSV,
#                      updates session state, spawns beep + push threads
#    show_alert_banner() — renders the red banner in the main page
# ══════════════════════════════════════════════════════════════

def _beep():
    """Play 4 short beeps on Windows to attract operator attention."""
    if WINSOUND_OK:
        for _ in range(4):
            winsound.Beep(1500, 250)

def send_push(msg):
    """Send a push notification via ntfy.sh (free, no account needed).
    Runs in a daemon thread so it never blocks the main Streamlit loop.
    """
    try:
        requests.post("https://ntfy.sh/smart_traffic_v3",
                      data=msg.encode(), timeout=2)
    except Exception:
        pass   # silently ignore network failures

def trigger_alert(event, plate="", lane=-1):
    """Record a traffic violation/event and broadcast alerts.

    Parameters
    ----------
    event : str   Human-readable event label (matches FINE_TABLE keys).
    plate : str   Number plate string if captured by OCR; else empty.
    lane  : int   0-based lane index; -1 means unknown/not applicable.

    Throttle: alerts fire at most once every 4 seconds to prevent spam.
    Ambulance events are automatically exempted from fines.
    """
    now = time.time()
    # ── Throttle: skip if last alert was < 4 seconds ago ──
    if now - st.session_state.last_alert_time < 4:
        return
    st.session_state.last_alert_time = now
    st.session_state.alert_active    = True
    st.session_state.alert_time      = now
    st.session_state.current_alert   = event

    # Ambulances are exempt from fines
    is_ambulance_event = any(k in event for k in ["Ambulance", "Siren", "🚑"])
    fine = 0 if is_ambulance_event else FINE_TABLE.get(event, 500)
    if not is_ambulance_event:
        st.session_state.total_fines += fine

    # ── Append violation record to session list and CSV ──
    st.session_state.violations.append({
        "Time":     datetime.now().strftime("%H:%M:%S"),
        "Event":    event,
        "Plate":    plate if plate else "—",
        "Lane":     f"Lane {lane+1}" if lane >= 0 else "—",
        "Fine(Rs)": fine,
        "Status":   "EXEMPT — Emergency Vehicle" if is_ambulance_event else "Pending",
    })
    try:
        with open(FINE_LOG, "a", newline="") as f:
            csv.writer(f).writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, plate, fine])
    except Exception:
        pass

    # ── Fire audio + push notification in background threads ──
    threading.Thread(target=_beep, daemon=True).start()
    threading.Thread(target=send_push,
                     args=(f"ALERT: {event} | Plate:{plate} | Rs{fine}",),
                     daemon=True).start()

def show_alert_banner():
    """Render the red flashing alert banner at the top of the main page.
    The banner auto-hides 12 seconds after the alert was triggered.
    """
    if not st.session_state.alert_active:
        return
    if time.time() - st.session_state.alert_time > 12:
        st.session_state.alert_active = False
        return
    msg = st.session_state.current_alert
    # Map event keywords to representative icons
    icon_map = {
        "Ambulance": "🚑", "Siren": "🚑", "Red Light": "🚨",
        "Overspeed": "🚗", "Helmet": "🪖", "Mobile": "📵", "Accident": "💥",
    }
    icon = next((v for k, v in icon_map.items() if k in msg), "⚠️")
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#8B0000,#cc0000);
                padding:16px 24px;border-radius:10px;text-align:center;
                font-size:22px;font-weight:700;color:#fff;
                box-shadow:0 0 30px #ff0000aa;margin-bottom:12px;
                border:1px solid #ff4444;">
      {icon}&nbsp;&nbsp;{msg}&nbsp;&nbsp;{icon}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  WEATHER
#  Fetches current temperature and weather code from Open-Meteo
#  (free, no API key required). Returns a formatted emoji string.
#  Used in the sidebar and previously in the weather-adapt panel.
#  Coordinates are looked up from LOCATION_COORDS by city name.
# ══════════════════════════════════════════════════════════════

def get_weather(city):
    lat, lon = LOCATION_COORDS.get(city, (13.08, 80.27))
    try:
        r  = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true", timeout=4)
        cw = r.json().get("current_weather", {})
        temp  = cw.get("temperature", "?")
        wcode = cw.get("weathercode", 0)
        icon  = "☀️" if wcode < 3 else "🌤" if wcode < 50 else "🌧" if wcode < 70 else "⛈"
        return f"{icon} {temp} °C"
    except Exception:
        return "🌡 N/A"

# ══════════════════════════════════════════════════════════════
#  SIREN DETECTION
#  Records 1 second of audio from the default microphone and
#  measures RMS volume. If volume is in the 0.04–0.6 range it
#  is likely a siren (quiet < 0.04 = background noise;
#  > 0.6 = clipping / construction noise).
#  Returns False immediately if sounddevice is unavailable.
# ══════════════════════════════════════════════════════════════

def detect_siren():
    if not AUDIO_AVAILABLE:
        return False
    try:
        audio = sd.rec(int(44100), samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        vol = float(np.linalg.norm(audio))
        return 0.04 < vol < 0.6
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════
#  AMBULANCE COLOUR HEURISTIC
#  Checks a vehicle ROI (region of interest) for the white body
#  and red cross/stripe colours characteristic of Indian ambulances.
#  Thresholds tuned for standard government ambulance livery.
#  Returns True if the ROI is likely an ambulance.
# ══════════════════════════════════════════════════════════════

def is_ambulance_roi(roi):
    if roi is None or roi.size < 400:
        return False
    try:
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        wht  = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        red1 = cv2.inRange(hsv, (0, 120, 70),  (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red  = cv2.bitwise_or(red1, red2)
        w_pct = np.count_nonzero(wht) / roi.size
        r_pct = np.count_nonzero(red) / roi.size
        return w_pct > 0.35 or r_pct > 0.04
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════
#  HELMET HEURISTIC
#  Checks the top 35% of a motorcycle detection ROI for skin-tone
#  pixels. High skin percentage → bare head → no helmet.
#  Returns True (has helmet) or False (possible violation).
#  False-positives are common in poor lighting; threshold = 0.55.
# ══════════════════════════════════════════════════════════════

def check_helmet(head_roi):
    if head_roi is None or head_roi.size < 100:
        return True
    try:
        hsv      = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        skin     = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
        skin_pct = np.count_nonzero(skin) / max(head_roi.size, 1)
        return skin_pct < 0.55
    except Exception:
        return True

# ══════════════════════════════════════════════════════════════
#  MODEL LOADERS
#  Both functions are decorated with @st.cache_resource so the
#  heavy models are loaded ONCE per Python process and reused
#  across all Streamlit re-runs (avoids ~1-2 s startup penalty).
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load YOLOv8n object detection model. Downloads on first run."""
    return YOLO(MODEL_NAME)

@st.cache_resource
def load_ocr():
    """Load EasyOCR English reader (CPU mode). Used for number plate OCR."""
    return easyocr.Reader(['en'], gpu=False, verbose=False)

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG + CSS
#  Sets the Streamlit page title, icon, and layout to "wide"
#  so all 4 lane columns have enough horizontal space.
#  The <style> block injects custom Google Fonts (Orbitron for
#  the futuristic headings, Inter for body text) and defines
#  reusable CSS classes used throughout the rest of the page:
#    .card        — dark gradient info card with hover glow
#    .alert-card  — red emergency card in sidebar
#    .page-title  — large centred Orbitron heading
#    .sig-box     — traffic signal widget container
#    .divider     — thin horizontal rule between sections
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Traffic Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background-color:#080c14;}
section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e2d40;}
.sidebar-logo{font-family:'Orbitron',monospace;font-size:17px;font-weight:700;color:#00ffcc;
              text-align:center;padding:12px 0 4px;letter-spacing:2px;}
.sidebar-sub{color:#3a5a7a;font-size:11px;text-align:center;margin-bottom:12px;}
.sb-section{background:#111827;border-radius:6px;padding:8px 12px;margin:8px 0;
            border-left:3px solid #00ffcc;font-size:11px;color:#557;
            font-weight:700;letter-spacing:1px;text-transform:uppercase;}
.card{background:linear-gradient(135deg,#0f1923 0%,#1a2535 100%);
      border:1px solid #1e3a5f;padding:16px 12px;border-radius:12px;
      text-align:center;margin-bottom:8px;
      box-shadow:0 4px 20px rgba(0,255,204,0.06);}
.card:hover{box-shadow:0 6px 24px rgba(0,255,204,0.14);}
.card-val{font-family:'Orbitron',monospace;font-size:28px;color:#00ffcc;font-weight:700;}
.card-lbl{color:#4a6a8a;font-size:11px;margin-top:4px;text-transform:uppercase;letter-spacing:1px;}
.alert-card{background:linear-gradient(90deg,#4a0000,#800000);border:1px solid #ff3333;
            padding:12px 18px;border-radius:8px;color:white;font-weight:700;font-size:15px;
            box-shadow:0 0 18px #ff000055;margin:5px 0;}
.page-title{font-family:'Orbitron',monospace;font-size:26px;color:#00ffcc;
            text-align:center;letter-spacing:4px;padding:10px 0 2px;
            text-shadow:0 0 20px #00ffcc55;}
.page-sub{color:#3a5a7a;font-size:13px;text-align:center;margin-bottom:14px;}
.sig-box{background:#0a0f1a;border:1px solid #1e2d40;border-radius:12px;
         padding:14px 18px;text-align:center;min-width:100px;}
.sig-lane{font-family:'Orbitron',monospace;font-size:12px;color:#00ffcc;}
.sig-count{color:#4a6a8a;font-size:11px;margin:3px 0;}
.divider{border-top:1px solid #1a2535;margin:16px 0;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD RESOURCES
#  Called once per session (cached). Initialises:
#    model      — YOLOv8n for vehicle + object detection
#    reader     — EasyOCR for number plate text recognition
#    controller — TrafficSignalController state machine
#  The controller is stored in session_state so its phase/timer
#  persist across Streamlit re-runs within the same session.
# ══════════════════════════════════════════════════════════════

model  = load_model()
reader = load_ocr()

if st.session_state.controller is None:
    st.session_state.controller = TrafficSignalController()
controller = st.session_state.controller

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
#  The sidebar is rendered inside a `with st.sidebar:` block.
#  It contains (top → bottom):
#    • Logo and version header
#    • Smart Location dropdowns (State → City → Area → Junction)
#      — LANES is derived from the selected junction
#    • Live OpenStreetMap mini-map iframe centred on the city
#    • Camera Source toggle (Upload Videos vs CCTV / IP Camera)
#      — CCTV mode shows per-lane URL text inputs
#    • AI Settings sliders (confidence, speed limit, OCR freq)
#    • Detection Module toggles (emergency, helmet, mobile, etc.)
#    • Live Weather widget (fetches Open-Meteo API)
#    • Active alert banner (mirrors main page alert)
#    • Session Stats grid (vehicles, violations, fines, plates)
#    • Reset System button (clears all session state except location)
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🚦 TRAFFIC AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Smart City Control System v3.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">📍 Smart Location</div>', unsafe_allow_html=True)
    state    = st.selectbox("State",    list(LOCATIONS.keys()), key="sel_state")
    city     = st.selectbox("City",     list(LOCATIONS[state].keys()), key="sel_city")
    area     = st.selectbox("Area",     list(LOCATIONS[state][city].keys()), key="sel_area")
    junction = st.selectbox("Junction", LOCATIONS[state][city][area], key="sel_junc")
    LANES    = JUNCTION_LANES.get(junction, 4)

    # Interactive live mini-map with junction marker
    lat, lon = LOCATION_COORDS.get(city, (13.08, 80.27))
    map_html = f"""
    <div style="border-radius:8px;overflow:hidden;border:1px solid #1e3a5f;margin-top:6px;">
      <iframe
        src="https://www.openstreetmap.org/export/embed.html?bbox={lon-0.03}%2C{lat-0.02}%2C{lon+0.03}%2C{lat+0.02}&layer=mapnik&marker={lat}%2C{lon}"
        style="width:100%;height:180px;border:0;" title="Junction Map">
      </iframe>
    </div>
    <div style="font-size:10px;color:#2a5a7a;text-align:right;margin-top:2px;">
      📡 {junction} &nbsp;·&nbsp; {lat:.4f}°N, {lon:.4f}°E
    </div>
    """
    components.html(map_html, height=200)

    st.markdown('<div class="sb-section">📹 Camera Source</div>', unsafe_allow_html=True)
    camera_mode = st.radio("Mode", ["Upload Videos", "CCTV / IP Camera"],
                           key="cam_mode", label_visibility="collapsed")

    cctv_inputs = []
    if camera_mode == "CCTV / IP Camera":
        st.caption("Enter RTSP/HTTP/device URL per lane")
        for i in range(LANES):
            url = st.text_input(
                f"Lane {i+1} URL",
                value=CCTV_URLS[i] if i < len(CCTV_URLS) else "",
                placeholder="rtsp://user:pass@192.168.x.x/stream",
                key=f"cctv_{i}")
            cctv_inputs.append(url)

    st.markdown('<div class="sb-section">🧠 AI Settings</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Detection Confidence", 0.25, 0.85, 0.40, 0.05)
    speed_limit = st.slider("Speed Limit (km/h)", 20, 120, SPEED_LIMIT, 5)
    ocr_every   = st.slider("OCR Every N Frames",  30, 120, 60, 10)

    st.markdown('<div class="sb-section">🚨 Detection Modules</div>', unsafe_allow_html=True)
    emergency    = st.toggle("🚑 Emergency / Ambulance Mode",  key="emg_tog")
    helmet_check = st.toggle("🪖 Helmet Violation Detection",   value=True)
    mobile_check = st.toggle("📵 Mobile Usage Detection",       value=True)
    rl_check     = st.toggle("🚨 Red Light Violation",          value=True)
    speed_check  = st.toggle("🚗 Overspeed Detection",          value=True)

    st.markdown('<div class="sb-section">🌦 Live Weather</div>', unsafe_allow_html=True)
    wx = get_weather(city)
    st.markdown(
        f'<div style="background:#111827;border-radius:8px;padding:10px;'
        f'text-align:center;color:#00ffcc;font-size:20px;font-weight:700;">{wx}</div>',
        unsafe_allow_html=True)

    if st.session_state.alert_active:
        msg  = st.session_state.current_alert
        imap = {"Ambulance":"🚑","Siren":"🚑","Red Light":"🚨",
                "Overspeed":"🚗","Helmet":"🪖","Mobile":"📵","Accident":"💥"}
        ico  = next((v for k, v in imap.items() if k in msg), "⚠️")
        st.markdown(f'<div class="alert-card">{ico} {msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">📊 Session Stats</div>', unsafe_allow_html=True)
    total_v_sb = sum(st.session_state.lane_counts)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;">
      <div style="background:#111827;border-radius:6px;padding:8px;text-align:center;">
        <div style="color:#00ffcc;font-weight:700;font-size:18px;">{total_v_sb}</div>
        <div style="color:#334;font-size:10px;">VEHICLES</div></div>
      <div style="background:#111827;border-radius:6px;padding:8px;text-align:center;">
        <div style="color:#ff5555;font-weight:700;font-size:18px;">{len(st.session_state.violations)}</div>
        <div style="color:#334;font-size:10px;">VIOLATIONS</div></div>
      <div style="background:#111827;border-radius:6px;padding:8px;text-align:center;">
        <div style="color:#ffcc00;font-weight:700;font-size:18px;">Rs{st.session_state.total_fines}</div>
        <div style="color:#334;font-size:10px;">FINES</div></div>
      <div style="background:#111827;border-radius:6px;padding:8px;text-align:center;">
        <div style="color:#88aaff;font-weight:700;font-size:18px;">{len(st.session_state.detected_plates)}</div>
        <div style="color:#334;font-size:10px;">PLATES</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔄 Reset System", use_container_width=True):
        keep = {"sel_state","sel_city","sel_area","sel_junc","cam_mode","emg_tog"}
        for k in [k for k in st.session_state if k not in keep]:
            del st.session_state[k]
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  MAIN PAGE — TITLE
#  Renders the large "SMART CITY TRAFFIC CONTROL" heading using
#  the .page-title CSS class, followed by the junction + location
#  sub-heading and the active alert banner (if any).
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="page-title">🚦 SMART CITY TRAFFIC CONTROL</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-sub">{junction} &nbsp;|&nbsp; {area}, {city} &nbsp;|&nbsp; {LANES} Lanes</div>',
            unsafe_allow_html=True)

show_alert_banner()

# ══════════════════════════════════════════════════════════════
#  SIREN CHECK (runs every 8 seconds to detect ambulance)
#  Uses microphone audio if sounddevice is available
# ══════════════════════════════════════════════════════════════
if time.time() - st.session_state.last_siren_check > 8:
    st.session_state.last_siren_check = time.time()
    if detect_siren():
        st.session_state.siren_active = True
        trigger_alert("🚑 Ambulance Siren Detected", lane=st.session_state.ambulance_lane)
    else:
        st.session_state.siren_active = False

# ── Emergency toggle from sidebar: force ambulance mode on Lane 1 ──
if emergency:
    st.session_state.ambulance_active = True
    st.session_state.ambulance_lane   = 0

# ══════════════════════════════════════════════════════════════
#  VIDEO INPUT
#  Only shown before detection starts (video_started == False).
#  Two modes controlled by the sidebar camera_mode radio:
#
#  Upload Videos:
#    • Shows one st.file_uploader per lane.
#    • When ALL lanes have uploads, saves each to lane_N.mp4 on
#      disk, stores paths in session_state, and triggers rerun.
#
#  CCTV / IP Camera:
#    • "Connect" button validates entered URLs / falls back to
#      local lane_N.mp4 files for lanes with no URL.
#    • Paths stored the same way as upload mode.
# ══════════════════════════════════════════════════════════════

if not st.session_state.video_started:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if camera_mode == "Upload Videos":
        st.subheader("📂 Upload Lane Videos")
        up_cols = st.columns(LANES)
        vids = []
        for i in range(LANES):
            with up_cols[i]:
                v = st.file_uploader(f"Lane {i+1}",
                                     type=["mp4","avi","mpeg4","mov"],
                                     key=f"up_{i}")
                vids.append(v)
        if all(v is not None for v in vids):
            paths = []
            with st.spinner("Saving uploaded videos…"):
                for i, v in enumerate(vids):
                    p = f"lane_{i}.mp4"
                    with open(p, "wb") as fh:
                        fh.write(v.read())
                    paths.append(p)
            st.session_state.video_paths   = paths
            st.session_state.video_started = True
            st.success(f"✅ {LANES} videos loaded — starting detection")
            st.rerun()

    elif camera_mode == "CCTV / IP Camera":
        st.subheader("📡 CCTV / IP Camera")
        ca, cb = st.columns(2)
        with ca:
            if st.button("🔗 Connect to CCTV Cameras", use_container_width=True):
                paths, ok = [], True
                for i, url in enumerate(cctv_inputs[:LANES]):
                    u = url.strip()
                    if u:
                        paths.append(u)
                    else:
                        demo = f"lane_{i}.mp4"
                        if Path(demo).exists():
                            paths.append(demo)
                        else:
                            st.warning(f"Lane {i+1}: no URL / local file — skipping")
                            ok = False
                if paths:
                    st.session_state.video_paths   = paths
                    st.session_state.video_started = True
                    st.rerun()
        with cb:
            st.markdown("""
            <div style="background:#111827;border-radius:8px;padding:12px;
                        font-size:12px;color:#4a6a8a;">
            <b style="color:#00ffcc">Supported formats:</b><br>
            rtsp://user:pass@192.168.x.x/stream<br>
            http://ip:port/video<br>
            0 / 1 / 2 — local webcam index<br>
            /dev/video0 — Linux device
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PREDICTION TREND
#  Appends the current total vehicle count to traffic_history
#  (capped at 60 samples ≈ ~1 minute of history at ~1 fps).
#  This rolling history is used by the traffic trend line chart
#  and the 5-sample moving-average traffic prediction label.
# ══════════════════════════════════════════════════════════════

current_total = sum(st.session_state.lane_counts)
st.session_state.traffic_history.append(current_total)
st.session_state.traffic_history = st.session_state.traffic_history[-60:]

# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
#  Runs only when video_started == True and video_paths is set.
#  Architecture (one iteration = one Streamlit re-run):
#
#  1. Opens a cv2.VideoCapture per lane (resumes from saved pos).
#  2. Reads one frame per lane.
#  3. Every 2nd re-run (do_infer): runs YOLOv8 on a 416×416 resize.
#  4. Per detection box:
#       • Ambulance heuristic  → immediate green + alert
#       • Vehicle type tally   → vtype_counts for distribution chart
#       • Helmet check         → violation if bare head on motorcycle
#       • Mobile usage         → violation on cell phone detection
#       • OCR (every N frames) → plate text → detected_plates list
#       • DeepSORT tracking    → persistent track IDs across frames
#  5. Per track: speed estimation via pixel displacement / Δt.
#     Speed > speed_limit → overspeed violation.
#     Track crosses stop line → red-light violation if signal is RED.
#  6. After all lanes: updates TrafficSignalController, saves
#     lane_counts + prev_positions to session_state.
# ══════════════════════════════════════════════════════════════

if st.session_state.video_started and st.session_state.video_paths:

    num_lanes = len(st.session_state.video_paths)
    if controller.num_lanes != num_lanes:
        st.session_state.controller = TrafficSignalController(num_lanes)
        controller = st.session_state.controller

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("📹 Live Camera Feeds")

    cam_cols    = st.columns(num_lanes)
    frame_slots = [c.empty() for c in cam_cols]
    stat_cols   = st.columns(num_lanes)
    stat_slots  = [c.empty() for c in stat_cols]

    # Open caps
    caps = []
    for path in st.session_state.video_paths:
        try:
            src = int(path)
        except (ValueError, TypeError):
            src = path
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.get(f"pos_{path}", 0))
        caps.append(cap)

    lane_counts        = [0] * num_lanes
    bus_count          = 0
    vtype_counts       = {"car":0,"bus":0,"truck":0,"motorcycle":0,"bicycle":0}
    ambulance_detected = False
    ambulance_lane_det = -1
    prev_positions     = st.session_state.prev_positions

    st.session_state.frame_skip += 1
    do_infer = True  # Run inference every frame for accurate detection

    # Per-lane tracker pool
    if "_trackers" not in st.session_state:
        st.session_state["_trackers"] = [
            DeepSort(max_age=25, n_init=2, max_cosine_distance=0.35)
            for _ in range(num_lanes)
        ]
    trackers = st.session_state["_trackers"]
    while len(trackers) < num_lanes:
        trackers.append(DeepSort(max_age=25, n_init=2, max_cosine_distance=0.35))

    # ── Per lane ───────────────────────────────────────────────
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if frame is None:
            frame_slots[i].warning(f"⚠️ Lane {i+1}: feed lost")
            cap.release()
            continue

        st.session_state[f"pos_{st.session_state.video_paths[i]}"] = \
            int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        disp  = cv2.resize(frame, (640, 360))
        infer = cv2.resize(frame, (416, 416))
        sx, sy = 640/416, 360/416

        vehicle_count = 0
        detections    = []
        broke_for_amb = False

        if do_infer:
            results = model(infer, conf=conf_thresh, imgsz=320, verbose=False)

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cs    = float(box.conf[0])
                label = model.names[int(box.cls[0])].lower()
                if cs < conf_thresh:
                    continue

                dx1 = int(x1*sx); dy1 = int(y1*sy)
                dx2 = int(x2*sx); dy2 = int(y2*sy)
                roi = infer[int(y1):int(y2), int(x1):int(x2)]

                # ── AMBULANCE (colour heuristic, not raw "truck" label) ──
                if label in ["truck","car","bus","van"]:
                    if is_ambulance_roi(roi):
                        ambulance_detected  = True
                        ambulance_lane_det  = i
                        st.session_state.ambulance_active = True
                        st.session_state.ambulance_lane   = i
                        cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,0,255),3)
                        cv2.putText(disp,"AMBULANCE",(dx1,max(dy1-38,12)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
                        trigger_alert("🚑 Ambulance Detected — Priority Cleared", lane=i)
                        broke_for_amb = True
                        break   # immediate priority

                if label in vtype_counts:
                    vtype_counts[label] += 1
                if label in ["car","bus","truck","motorcycle","bicycle"]:
                    vehicle_count += 1
                if label == "bus":
                    bus_count += 2

                # ── HELMET CHECK ──
                if helmet_check and label == "motorcycle":
                    hy2 = int(y1 + (y2-y1)*0.35)
                    head_roi = infer[int(y1):hy2, int(x1):int(x2)]
                    if not check_helmet(head_roi):
                        trigger_alert("🪖 Helmet Violation", lane=i)
                        cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,165,255),3)
                        cv2.putText(disp,"NO HELMET",(dx1,max(dy1-30,10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,165,255),2)

                # ── MOBILE ──
                if mobile_check and label in ["cell phone","remote"] and cs > 0.45:
                    trigger_alert("📵 Mobile Usage", lane=i)

                # ── NUMBER PLATE OCR — runs in background thread to avoid blocking ──
                st.session_state.frame_count += 1
                plate_txt = ""
                if (label in ["car","motorcycle","truck","bus"]
                        and st.session_state.frame_count % ocr_every == 0
                        and roi.size > 200):
                    roi_copy = roi.copy()
                    def _run_ocr(r, dx1=dx1, dy2=dy2):
                        try:
                            with HiddenPrints():
                                ocr_res = reader.readtext(
                                    r,
                                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            for entry in ocr_res:
                                txt = entry[1].upper().replace(" ","")
                                if 5 < len(txt) < 13:
                                    if txt not in st.session_state.detected_plates:
                                        st.session_state.detected_plates.append(txt)
                        except Exception:
                            pass
                    threading.Thread(target=_run_ocr, args=(roi_copy,), daemon=True).start()

                # ── STANDARD BOX ──
                cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,255,180),2)
                cv2.putText(disp,f"{label} {cs:.2f}",
                            (dx1,max(dy1-6,10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,180),1)

                detections.append(
                    ([float(x1),float(y1),float(x2-x1),float(y2-y1)], cs, label))

            # ── DEEPSORT TRACKING ──
            if not broke_for_amb:
                try:
                    tracks = trackers[i].update_tracks(detections, frame=infer)
                    STOP_Y = int(360 * STOP_LINE_FRAC)
                    confirmed_track_count = 0  # Stable unique vehicle count via DeepSORT

                    for trk in tracks:
                        if not trk.is_confirmed():
                            continue
                        confirmed_track_count += 1  # Only count confirmed tracks
                        tid = trk.track_id
                        tl,tt,tr,tb = trk.to_ltrb()
                        tl = int(tl*sx); tt = int(tt*sy)
                        tr = int(tr*sx); tb = int(tb*sy)
                        cx = (tl+tr)//2; cy = (tt+tb)//2

                        # Speed
                        if tid in prev_positions:
                            px,py,pt = prev_positions[tid]
                            dist = np.sqrt((cx-px)**2+(cy-py)**2)
                            dt   = time.time()-pt
                            spd  = int((dist/dt)*PIXEL_TO_KMH) if dt > 0 else 0
                        else:
                            spd = 0
                        prev_positions[tid] = (cx, cy, time.time())

                        spd_col = (0,0,255) if spd > speed_limit else (0,200,255)
                        cv2.putText(disp,f"{spd}km/h",
                                    (tl, max(tt-10,10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.45,spd_col,1)
                        cv2.putText(disp,f"#{tid}",
                                    (tl, max(tt-24,10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,0),1)

                        # Red light violation
                        if rl_check and controller.get_light(i)=="RED" and cy > STOP_Y:
                            trigger_alert("🚨 Red Light Violation", lane=i)
                            cv2.putText(disp,"RED LIGHT CROSS!",
                                        (tl,max(tt-40,10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)

                        # Overspeed
                        if speed_check and spd > speed_limit:
                            trigger_alert("🚗 Overspeed", lane=i)
                            cv2.rectangle(disp,(tl,tt),(tr,tb),(0,0,255),3)

                    # Update vehicle_count with stable DeepSORT count
                    vehicle_count = confirmed_track_count
                except Exception:
                    pass

        # Stop line overlay
        STOP_Y    = int(360*STOP_LINE_FRAC)
        light_now = controller.get_light(i)
        lc = {"GREEN":(0,220,0),"YELLOW":(0,200,200),"RED":(0,0,220)}.get(light_now,(150,150,150))
        cv2.line(disp,(0,STOP_Y),(640,STOP_Y),lc,2)
        cv2.putText(disp,f"STOP [{light_now}]",(4,STOP_Y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,lc,1)

        # Lane overlay
        cv2.putText(disp,f"LANE {i+1}",(8,22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,204),2)
        cv2.putText(disp,f"Veh:{vehicle_count}",(8,46),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

        lane_counts[i] = vehicle_count
        frame_slots[i].image(disp, channels="BGR", use_container_width=True)
        stat_slots[i].metric(f"Lane {i+1}", f"{vehicle_count} vehicles")
        cap.release()

    # Save
    st.session_state.lane_counts    = lane_counts
    st.session_state.bus_count      = bus_count
    st.session_state.prev_positions = prev_positions

    # Signal update
    amb_lane = st.session_state.ambulance_lane if st.session_state.ambulance_active else -1
    controller.update(lane_counts, ambulance_lane=amb_lane,
                      siren=st.session_state.siren_active)

    # Clear ambulance after 20 s
    if st.session_state.ambulance_active:
        if time.time() - st.session_state.last_alert_time > 20:
            st.session_state.ambulance_active = False

    # Confidence / distribution
    total_v      = sum(lane_counts) + 1
    conf_score   = max(lane_counts) / total_v
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    ca2, cb2 = st.columns(2)
    with ca2:
        st.subheader("🤖 AI Confidence")
        st.metric("Model Confidence", f"{conf_score*100:.1f}%")
        best = lane_counts.index(min(lane_counts))
        st.success(f"✅ Least-congested Lane: Lane {best+1}")
    with cb2:
        st.subheader("🚗 Vehicle Distribution")
        if sum(vtype_counts.values()) > 0:
            vdf = pd.DataFrame.from_dict(vtype_counts, orient="index", columns=["count"])
            st.bar_chart(vdf, height=200)
        else:
            st.info("Awaiting detections…")

    # FPS
    now_t = time.time()
    fps   = 1/(now_t-st.session_state.prev_time) if (now_t-st.session_state.prev_time)>0 else 0
    st.session_state.prev_time = now_t
    st.metric("⚡ FPS", f"{fps:.1f}")

# ══════════════════════════════════════════════════════════════
#  SIGNAL PANEL
#  Renders one traffic-light widget per lane using inline HTML/CSS.
#  Each widget shows:
#    • Lane number and current vehicle count
#    • 🚑 PRIORITY badge if ambulance is active on that lane
#    • Three coloured circles (R/Y/G) with CSS glow on active one
#    • Current signal phase label (GREEN / YELLOW / RED)
#    • Countdown timer for active phase
#  Uses controller.get_light() and controller.time_remaining().
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🚦 Traffic Signal Status")

n_sig    = len(st.session_state.video_paths) if st.session_state.video_paths else LANES
sig_cols = st.columns(n_sig)

for i in range(n_sig):
    light = controller.get_light(i)
    cnt   = st.session_state.lane_counts[i] if i < len(st.session_state.lane_counts) else 0
    timer = controller.time_remaining(i)
    amb   = (i == st.session_state.ambulance_lane and st.session_state.ambulance_active)

    r = "#ff3333" if light=="RED"    else "#1a0000"
    y = "#ffdd00" if light=="YELLOW" else "#1a1200"
    g = "#00ee00" if light=="GREEN"  else "#001a00"
    rg = "0 0 16px #ff3333" if light=="RED"    else "none"
    yg = "0 0 16px #ffdd00" if light=="YELLOW" else "none"
    gg = "0 0 16px #00ee00" if light=="GREEN"  else "none"
    sc = "#00ee00" if light=="GREEN" else "#ff3333" if light=="RED" else "#ffdd00"
    amb_b = '<div style="color:#ff5555;font-size:11px;font-weight:700;margin:2px 0;">🚑 PRIORITY</div>' if amb else ""

    sig_cols[i].markdown(f"""
    <div class="sig-box">
      <div class="sig-lane">LANE {i+1}</div>
      <div class="sig-count">{cnt} veh</div>
      {amb_b}
      <div style="background:#050a10;border-radius:10px;padding:10px 8px;
                  width:66px;margin:8px auto;border:1px solid #1a2535;">
        <div style="width:26px;height:26px;background:{r};border-radius:50%;
                    margin:5px auto;box-shadow:{rg};"></div>
        <div style="width:26px;height:26px;background:{y};border-radius:50%;
                    margin:5px auto;box-shadow:{yg};"></div>
        <div style="width:26px;height:26px;background:{g};border-radius:50%;
                    margin:5px auto;box-shadow:{gg};"></div>
      </div>
      <div style="font-family:'Orbitron',monospace;font-size:13px;
                  font-weight:700;color:{sc};">{light}</div>
      <div style="color:#2a4a6a;font-size:11px;margin-top:3px;">⏱ {timer}s</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONGESTION METRICS
#  Computes a weighted traffic score:
#    t_score = Σ(vehicles × VEHICLE_WEIGHT) + buses × BUS_WEIGHT
#  Displays a colour-coded congestion banner and 5 KPI cards:
#    Total Vehicles | Buses | Active Lanes | Violations | Total Fines
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
lane_counts  = st.session_state.lane_counts
bus_count    = st.session_state.bus_count
total_v      = sum(lane_counts)
t_score      = total_v*VEHICLE_WEIGHT + bus_count*BUS_WEIGHT

st.subheader("🚦 Real-Time Congestion")
if t_score < 10:
    st.success("🟢 Low Traffic — All lanes flowing normally")
elif t_score < 25:
    st.warning("🟡 Moderate Traffic — Monitor signals")
else:
    st.error("🔴 HIGH CONGESTION — Extended green assigned to busiest lane")

m1,m2,m3,m4,m5 = st.columns(5)
for col,(val,lbl,clr) in zip(
    [m1,m2,m3,m4,m5],
    [(total_v,          "Total Vehicles",  "#00ffcc"),
     (bus_count//2,     "Buses",           "#ffaa00"),
     (LANES,            "Active Lanes",    "#88aaff"),
     (len(st.session_state.violations), "Violations", "#ff5555"),
     (f"Rs{st.session_state.total_fines}","Total Fines","#ffdd00")]):
    col.markdown(
        f'<div class="card"><div class="card-val" style="color:{clr};">{val}</div>'
        f'<div class="card-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CHARTS
#  Three visualisations rendered side-by-side then full-width:
#
#  Lane Traffic Density (bar chart)
#    — Current vehicle count per lane as a simple bar chart.
#
#  Traffic Trend (line chart)
#    — Rolling 60-sample history of total vehicle count.
#      Shows congestion build-up and clearance over time.
#
#  Live Congestion Heatmap (plotly imshow)
#    — 2D matrix: rows = time ticks (last 10), columns = lanes.
#    — Colour scale RdYlGn_r: red = congested, green = clear.
#    — Helps operators spot which lane/time combination
#      is repeatedly congested.
#
#  Lane Congestion Progress Bars
#    — Per-lane mini-cards with a proportional fill bar and
#      LOW / MED / HIGH label beneath the heatmap.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
ch1, ch2 = st.columns(2)
with ch1:
    st.subheader("📊 Lane Traffic Density")
    tdf = pd.DataFrame({
        "Lane":     [f"L{i+1}" for i in range(len(lane_counts))],
        "Vehicles": lane_counts})
    st.bar_chart(tdf.set_index("Lane"), height=260)
with ch2:
    st.subheader("📈 Traffic Trend")
    hist = st.session_state.traffic_history
    st.line_chart(pd.DataFrame({"Vehicles":hist}), height=260)

st.subheader("🔥 Live Congestion Heatmap")

# Build a richer heatmap: time-series rows + current lane columns
hist_data = st.session_state.traffic_history[-10:]  # last 10 ticks
if len(hist_data) < 2:
    # Not enough history yet — show current lanes only as a single row
    hmap_matrix = np.array(lane_counts, dtype=float).reshape(1, -1)
    row_labels  = ["Now"]
else:
    # Simulate per-lane history by distributing total proportionally to current distribution
    n_rows = min(len(hist_data), 10)
    total_now = sum(lane_counts) + 1
    weights   = [c / total_now for c in lane_counts] if total_now > 0 else [0.25] * len(lane_counts)
    hmap_matrix = np.array(
        [[round(hist_data[-(n_rows - r)] * w) for w in weights] for r in range(n_rows)],
        dtype=float)
    row_labels = [f"T-{n_rows - r - 1}" if r < n_rows - 1 else "Now" for r in range(n_rows)]

col_labels = [f"Lane {i+1}" for i in range(len(lane_counts))]
hdf2 = pd.DataFrame(hmap_matrix, columns=col_labels, index=row_labels)

fig2 = px_plot.imshow(
    hdf2,
    text_auto=True,
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    labels={"x": "Lane", "y": "Time", "color": "Vehicles"},
    zmin=0, zmax=max(MAX_GREEN, hmap_matrix.max() + 1),
)
fig2.update_layout(
    height=280,
    margin=dict(l=40, r=20, t=20, b=40),
    paper_bgcolor="#080c14",
    plot_bgcolor="#080c14",
    font_color="#aaaaaa",
    coloraxis_colorbar=dict(
        tickfont=dict(color="#aaaaaa"),
        title=dict(text="Veh", font=dict(color="#aaaaaa")),
        thickness=12, len=0.8,
    ),
    xaxis=dict(tickfont=dict(color="#00ffcc", size=11), gridcolor="#1a2535"),
    yaxis=dict(tickfont=dict(color="#aaaaaa", size=10),  gridcolor="#1a2535"),
)
fig2.update_traces(textfont=dict(color="white", size=11))
st.plotly_chart(fig2, use_container_width=True)

# Lane-by-lane congestion bars  
bar_cols = st.columns(len(lane_counts))
for i, (cnt, col) in enumerate(zip(lane_counts, bar_cols)):
    level = "🔴 HIGH" if cnt > 8 else "🟡 MED" if cnt > 4 else "🟢 LOW"
    clr   = "#ff4444" if cnt > 8 else "#ffcc00" if cnt > 4 else "#00ee88"
    pct   = min(int(cnt / max(max(lane_counts), 1) * 100), 100)
    col.markdown(
        f'<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:8px;'
        f'padding:8px 10px;text-align:center;">'
        f'<div style="font-family:monospace;font-size:11px;color:#00ffcc;">Lane {i+1}</div>'
        f'<div style="font-size:18px;font-weight:700;color:{clr};">{cnt}</div>'
        f'<div style="background:#111827;border-radius:4px;height:6px;margin:4px 0;">'
        f'<div style="background:{clr};width:{pct}%;height:6px;border-radius:4px;'
        f'transition:width 0.4s;"></div></div>'
        f'<div style="font-size:9px;color:#334;">{level}</div>'
        f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  NEW FEATURE 1 — LANE-WISE CO₂ EMISSION ESTIMATOR
#  ─────────────────────────────────────────────────────────────
#  Estimates real-time CO₂ emissions per lane using vehicle
#  counts, vehicle-type weights, and idle/moving emission rates.
#  Emission model:
#    • Car idle  = 2.6 g CO₂/min  |  moving = 1.8 g/min
#    • Truck/Bus = 3× car factor
#    • Motorcycle= 0.6× car factor
#  Displays per-lane emission gauge + total junction output.
#  Section runs from here → END OF CO₂ FEATURE
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🌿 Lane-wise CO₂ Emission Estimator")

# ── Emission constants (grams of CO₂ per minute per vehicle) ──
CO2_IDLE_CAR     = 2.6    # car idling at red light
CO2_MOVE_CAR     = 1.8    # car moving through green
CO2_HEAVY_FACTOR = 3.0    # multiplier for bus/truck
CO2_MOTO_FACTOR  = 0.6    # multiplier for motorcycle

# ── Per-lane emission calculation ──
# Determines if lane is mostly idle (red) or moving (green)
lane_co2 = []
for idx, cnt in enumerate(lane_counts):
    # Check the current signal state for this lane
    lane_signal = controller.get_light(idx)
    # Heavy vehicle estimate: roughly 15% of lane count are buses/trucks
    heavy_est   = max(1, int(cnt * 0.15))
    moto_est    = max(0, int(cnt * 0.20))
    car_est     = cnt - heavy_est - moto_est

    # If lane is RED, vehicles are idling → higher emissions
    if lane_signal == "RED":
        em = (car_est * CO2_IDLE_CAR
              + heavy_est * CO2_IDLE_CAR * CO2_HEAVY_FACTOR
              + moto_est  * CO2_IDLE_CAR * CO2_MOTO_FACTOR)
    else:  # GREEN or YELLOW — vehicles are moving
        em = (car_est * CO2_MOVE_CAR
              + heavy_est * CO2_MOVE_CAR * CO2_HEAVY_FACTOR
              + moto_est  * CO2_MOVE_CAR * CO2_MOTO_FACTOR)
    lane_co2.append(round(em, 1))

# ── Render one card per lane ──
co2_cols = st.columns(len(lane_counts))
for idx, (co2_val, co2_col) in enumerate(zip(lane_co2, co2_cols)):
    # Choose colour based on emission level thresholds
    co2_clr = ("#ff4444" if co2_val > 15
               else "#ffcc00" if co2_val > 7
               else "#00ee88")
    co2_lbl = "🔴 High" if co2_val > 15 else "🟡 Medium" if co2_val > 7 else "🟢 Low"
    co2_col.markdown(
        f'<div class="card" style="min-height:110px;">'
        f'<div style="font-size:20px;">🏭</div>'
        f'<div style="font-family:\'Orbitron\',monospace;font-size:10px;'
        f'color:#00ffcc;margin:4px 0;">LANE {idx+1} CO₂</div>'
        f'<div style="font-size:20px;font-weight:700;color:{co2_clr};">{co2_val} g/min</div>'
        f'<div style="font-size:10px;color:#4a6a8a;">{co2_lbl} emission</div>'
        f'</div>', unsafe_allow_html=True)

# ── Total junction CO₂ summary row ──
total_co2     = round(sum(lane_co2), 1)
hourly_co2_kg = round(total_co2 * 60 / 1000, 2)   # convert g/min → kg/hr
co2_status_clr = "#ff4444" if total_co2 > 40 else "#ffcc00" if total_co2 > 20 else "#00ee88"
st.markdown(
    f'<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:8px;'
    f'padding:10px 16px;margin-top:8px;display:flex;justify-content:space-between;'
    f'align-items:center;">'
    f'<span style="color:#4a6a8a;font-size:12px;">Junction Total</span>'
    f'<span style="font-size:18px;font-weight:700;color:{co2_status_clr};">'
    f'{total_co2} g/min &nbsp;≈&nbsp; {hourly_co2_kg} kg/hr</span>'
    f'<span style="font-size:10px;color:#2a6a4a;">🌱 Green signal reduces ~30% idle emissions</span>'
    f'</div>', unsafe_allow_html=True)
# ── END OF CO₂ FEATURE ──

# ══════════════════════════════════════════════════════════════
#  NEW FEATURE 2 — JUNCTION INCIDENT HEATMAP REPORTER
#  ─────────────────────────────────────────────────────────────
#  Maintains a live incident log per lane using session state.
#  Categorises incidents (Violation, Congestion, Ambulance) with
#  timestamps; renders an interactive plotly timeline heatmap so
#  operators can spot recurring hot-spots at a glance.
#  Section runs from here → END OF INCIDENT HEATMAP
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🗺️ Junction Incident Heatmap Reporter")

# ── Initialise persistent incident store in session state ──
if "incident_log" not in st.session_state:
    st.session_state["incident_log"] = []   # list of {lane, category, time_label}

# ── Append new incidents from the current frame ──
# Each violation that was just recorded gets mapped to a lane + category
current_time_label = datetime.now().strftime("%H:%M")
for viol in st.session_state.violations[-5:]:    # check last 5 to catch new ones
    lane_str  = viol.get("Lane", "Lane 1")
    lane_idx  = int(lane_str.replace("Lane ", "").strip()) - 1 if lane_str != "—" else 0
    ev        = viol.get("Event", "")
    # Classify incident category for the heatmap
    if "Ambulance" in ev or "Siren" in ev:
        cat = "Ambulance"
    elif "Congestion" in ev or lane_counts[lane_idx] > 8:
        cat = "Congestion"
    else:
        cat = "Violation"

    # Avoid duplicating entries within the same minute
    entry = {"lane": lane_idx + 1, "category": cat, "time": current_time_label}
    if entry not in st.session_state["incident_log"]:
        st.session_state["incident_log"].append(entry)

# Also log high-congestion lanes even without a formal violation
for idx, cnt in enumerate(lane_counts):
    if cnt > 8:
        cong_entry = {"lane": idx + 1, "category": "Congestion", "time": current_time_label}
        if cong_entry not in st.session_state["incident_log"]:
            st.session_state["incident_log"].append(cong_entry)

# ── Keep only the last 60 incidents to avoid unbounded growth ──
st.session_state["incident_log"] = st.session_state["incident_log"][-60:]

# ── Build matrix: rows = lanes, columns = incident categories ──
inc_log = st.session_state["incident_log"]
cat_list = ["Violation", "Congestion", "Ambulance"]
lane_list = [f"Lane {i+1}" for i in range(LANES)]

# Count incidents per (lane, category) pair
inc_matrix = {cat: [0]*LANES for cat in cat_list}
for entry in inc_log:
    ln  = entry["lane"] - 1    # convert to 0-based index
    cat = entry["category"]
    if 0 <= ln < LANES and cat in inc_matrix:
        inc_matrix[cat][ln] += 1

# Build a DataFrame suitable for plotly imshow
inc_df = pd.DataFrame(inc_matrix, index=lane_list)

# ── Plot the incident heatmap ──
inc_fig = px_plot.imshow(
    inc_df.T,                   # transpose so lanes are on X axis
    text_auto=True,
    color_continuous_scale="YlOrRd",
    aspect="auto",
    labels={"x": "Lane", "y": "Incident Type", "color": "Count"},
    zmin=0,
)
inc_fig.update_layout(
    height=220,
    margin=dict(l=60, r=20, t=20, b=40),
    paper_bgcolor="#080c14",
    plot_bgcolor="#080c14",
    font_color="#aaaaaa",
    coloraxis_colorbar=dict(
        tickfont=dict(color="#aaaaaa"),
        title=dict(text="Count", font=dict(color="#aaaaaa")),
        thickness=12, len=0.8,
    ),
    xaxis=dict(tickfont=dict(color="#00ffcc", size=11), gridcolor="#1a2535"),
    yaxis=dict(tickfont=dict(color="#aaaaaa", size=10),  gridcolor="#1a2535"),
)
inc_fig.update_traces(textfont=dict(color="white", size=12))
st.plotly_chart(inc_fig, use_container_width=True)

# ── Recent incidents list — last 8 entries ──
if inc_log:
    st.caption("🕐 Recent incidents (latest first):")
    recent_cols = st.columns(4)
    for j, entry in enumerate(reversed(inc_log[-8:])):
        cat_icon = {"Violation":"🚨","Congestion":"🟠","Ambulance":"🚑"}.get(entry["category"],"⚠️")
        recent_cols[j % 4].markdown(
            f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:6px;'
            f'padding:5px 8px;font-size:11px;color:#aaaaaa;">'
            f'{cat_icon} <b style="color:#00ffcc">Lane {entry["lane"]}</b> '
            f'· {entry["category"]} · {entry["time"]}</div>',
            unsafe_allow_html=True)
else:
    st.info("✅ No incidents recorded yet this session.")
# ── END OF INCIDENT HEATMAP ──

# ══════════════════════════════════════════════════════════════
#  NEW FEATURE 3 — GREEN WAVE CORRIDOR OPTIMIZER
#  ─────────────────────────────────────────────────────────────
#  Calculates and displays the optimal "green wave" sequence —
#  the ideal lane order and timing offsets that allow vehicles
#  to travel through multiple consecutive junctions without
#  stopping, based on current lane density and signal timing.
#  Displays recommended sequence, offset table, and estimated
#  fuel & time savings for the current junction state.
#  Section runs from here → END OF GREEN WAVE FEATURE
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🌊 Green Wave Corridor Optimizer")

# ── Sort lanes by ascending vehicle count (least → most congested)
#    This gives the optimum dispatch order: clear lighter lanes first
#    so downstream junctions stay clear for the heavier ones.
sorted_lanes = sorted(range(len(lane_counts)), key=lambda x: lane_counts[x])

# ── Compute recommended green-time offset for each lane ──
#    Offset = cumulative green time of all preceding lanes + yellow gaps
#    Assumption: average inter-junction distance 200 m, avg speed 40 km/h
#    → travel time per junction ≈ 18 seconds
INTER_JUNCTION_TRAVEL_S = 18    # seconds for vehicle to reach next junction
YELLOW_GAP_S            = 3     # yellow buffer between phases

gw_rows = []           # collects data for the summary table
cumulative_offset = 0  # running offset in seconds

for rank, lane_idx in enumerate(sorted_lanes):
    cnt         = lane_counts[lane_idx]
    green_t     = controller.green_times[lane_idx] if lane_idx < len(controller.green_times) else BASE_GREEN
    offset_s    = cumulative_offset
    # Estimated vehicles that can clear junction in this green phase
    # Throughput model: ~1 vehicle per 2 s at green
    clearance   = min(int(green_t / 2), cnt)
    saved_stops = clearance            # each cleared vehicle avoids 1 stop
    fuel_saved  = round(saved_stops * 0.04, 2)   # ~40 mL fuel per stop avoidance
    time_saved  = round(saved_stops * 12 / 60, 1) # ~12s delay per stop

    gw_rows.append({
        "Lane":           f"Lane {lane_idx + 1}",
        "Rank":           rank + 1,
        "Vehicles":       cnt,
        "Green Time (s)": int(green_t),
        "Offset (s)":     offset_s,
        "Clearance Est.": clearance,
        "Fuel Saved (L)": fuel_saved,
        "Time Saved (m)": time_saved,
    })
    # Next lane's offset = this lane's green + yellow gap + travel time
    cumulative_offset += int(green_t) + YELLOW_GAP_S + INTER_JUNCTION_TRAVEL_S

# ── Render the optimised sequence header ──
seq_str = " → ".join([f"Lane {sorted_lanes[r]+1}" for r in range(len(sorted_lanes))])
total_fuel_saved = round(sum(r["Fuel Saved (L)"] for r in gw_rows), 2)
total_time_saved = round(sum(r["Time Saved (m)"] for r in gw_rows), 1)

st.markdown(
    f'<div style="background:linear-gradient(135deg,#071a2c,#0d2940);'
    f'border:1px solid #00ffcc55;border-radius:10px;padding:12px 18px;margin-bottom:12px;">'
    f'<div style="font-family:\'Orbitron\',monospace;font-size:11px;color:#00ffcc;'
    f'letter-spacing:1px;margin-bottom:6px;">OPTIMAL WAVE SEQUENCE</div>'
    f'<div style="font-size:18px;font-weight:700;color:#aaffcc;">{seq_str}</div>'
    f'<div style="display:flex;gap:24px;margin-top:8px;">'
    f'<span style="font-size:12px;color:#ffcc00;">⛽ Fuel saved: ~{total_fuel_saved} L/cycle</span>'
    f'<span style="font-size:12px;color:#88aaff;">⏱ Time saved: ~{total_time_saved} min/cycle</span>'
    f'<span style="font-size:12px;color:#aaffaa;">🌿 Lower idle emissions per vehicle</span>'
    f'</div></div>', unsafe_allow_html=True)

# ── Display the wave offset table ──
gw_df = pd.DataFrame(gw_rows)
st.dataframe(
    gw_df,
    use_container_width=True,
    height=min(50 + len(gw_rows) * 38, 260),
    hide_index=True,
)

# ── Visual lane-by-lane wave timing bar ──
gw_bar_cols = st.columns(len(gw_rows))
for col, row in zip(gw_bar_cols, gw_rows):
    bar_clr = "#00ffcc" if row["Rank"] == 1 else "#88aaff" if row["Rank"] == 2 else "#4a6a8a"
    col.markdown(
        f'<div class="card" style="min-height:90px;padding:10px 8px;">'
        f'<div style="font-size:10px;font-family:\'Orbitron\',monospace;color:{bar_clr};">'
        f'{row["Lane"]}</div>'
        f'<div style="font-size:15px;font-weight:700;color:{bar_clr};">#{row["Rank"]}</div>'
        f'<div style="font-size:10px;color:#4a6a8a;">Offset: {row["Offset (s)"]}s</div>'
        f'<div style="font-size:10px;color:#ffaa00;">Green: {row["Green Time (s)"]}s</div>'
        f'</div>', unsafe_allow_html=True)
# ── END OF GREEN WAVE FEATURE ──

# ══════════════════════════════════════════════════════════════
#  NUMBER PLATES
#  Displays the unique plates captured by EasyOCR this session.
#  Plates are stored in session_state.detected_plates (list).
#  Shows up to 18 most recent plates in a 6-column grid.
#  Plate cards use monospace font on a dark background to mimic
#  the appearance of a real Indian number plate display.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🔢 Detected Number Plates")
plates = st.session_state.detected_plates
if plates:
    p_cols = st.columns(min(len(plates), 6))
    for j, p in enumerate(plates[-18:]):
        p_cols[j % 6].markdown(
            f'<div style="background:#0f1923;border:1px solid #1e3a5f;'
            f'border-radius:6px;padding:6px 10px;text-align:center;'
            f'font-family:monospace;color:#00ffcc;font-size:13px;">{p}</div>',
            unsafe_allow_html=True)
else:
    st.info("No plates detected yet — OCR runs every N frames on vehicles")

# ══════════════════════════════════════════════════════════════
#  VIOLATION / FINE LOG
#  Renders all recorded violations as an interactive dataframe.
#  Below the table: total fines (₹), violation count, and unique
#  plate count are shown as metric tiles.
#  A download button exports the full violation log as CSV for
#  offline reporting or submission to traffic authority.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🚓 Auto Fine & Violation Log")
viols = st.session_state.violations
if viols:
    vdf = pd.DataFrame(viols)
    st.dataframe(vdf, use_container_width=True, height=260)
    tc1, tc2, tc3 = st.columns(3)
    tc1.error(f"💰 Total Fines: Rs {st.session_state.total_fines}")
    tc2.warning(f"📋 Violations: {len(viols)}")
    tc3.info(f"🔢 Unique Plates: "
             f"{len(set(r.get('Plate','') for r in viols if r.get('Plate','')!='—'))}")
    st.download_button(
        "⬇️ Download Violation Report (CSV)",
        vdf.to_csv(index=False).encode(),
        "violations.csv", "text/csv",
        use_container_width=True)
else:
    st.info("✅ No violations recorded yet")

# ══════════════════════════════════════════════════════════════
#  LOG APPEND
#  Appends one row to traffic_log.csv on every re-run.
#  Columns: timestamp, total vehicles, and per-lane counts.
#  The header row is written only if the file does not yet exist.
#  Wrapped in try/except to silently ignore any I/O errors
#  (e.g. read-only filesystem or disk full).
# ══════════════════════════════════════════════════════════════

try:
    lc_safe = (st.session_state.lane_counts + [0,0,0,0])[:4]
    pd.DataFrame([{
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total":   sum(lc_safe),
        "lane1":   lc_safe[0], "lane2": lc_safe[1],
        "lane3":   lc_safe[2], "lane4": lc_safe[3],
    }]).to_csv(LOG_FILE, mode="a",
               header=not Path(LOG_FILE).exists(), index=False)
except Exception:
    pass

# ══════════════════════════════════════════════════════════════
#  FOOTER
#  Simple centred attribution line at the bottom of the page.
#  Lists the key libraries and features for quick reference.
#  Uses a very dark colour (#1e2d40) so it is unobtrusive.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#1e2d40;font-size:11px;padding:6px 0;">
  Smart City Traffic Control v3.0 &nbsp;|&nbsp;
  YOLOv8 + DeepSORT + EasyOCR &nbsp;|&nbsp;
  Auto Fine · CCTV · Ambulance Priority · Helmet · Speed &nbsp;|&nbsp;
  CO₂ Estimator · Incident Heatmap · Green Wave Optimizer
</div>""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════
#  AUTO-REFRESH LOOP
#  Without this, Streamlit only redraws when the user clicks
#  something. This loop ensures signal timers and video feeds
#  update continuously at ~2 FPS.
# ══════════════════════════════════════════════════════════════
if st.session_state.video_started:
    time.sleep(0.4)   # ~2.5 re-runs per second
    st.rerun()
