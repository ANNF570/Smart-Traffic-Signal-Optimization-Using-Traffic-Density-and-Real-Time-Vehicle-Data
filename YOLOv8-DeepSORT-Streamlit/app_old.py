
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║   SMART CITY TRAFFIC CONTROL SYSTEM  v4.0                           ║
║   YOLOv8 + DeepSORT + EasyOCR + Streamlit                           ║
║   NEW: Ambulance Priority Flash | Bus Schedule Tracker               ║
║        Pedestrian Safety | Junction Intelligence | No-Refresh Upload ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, platform, threading, warnings, csv, pickle
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import plotly.express as px_plot
import streamlit as st
import streamlit.components.v1 as components

try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import easyocr

warnings.filterwarnings("ignore")

# ── MUST be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Smart Traffic Control v4.0",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# ══════════════════════════════════════════════════════════════

MODEL_NAME     = r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\runs\detect\runs\smart_traffic_fixed\weights\best.pt"
BASE_GREEN     = 10
MAX_GREEN      = 40
YELLOW_TIME    = 3
SPEED_LIMIT    = 60
STOP_LINE_FRAC = 0.62
PIXEL_TO_KMH   = 0.08
BUS_WEIGHT     = 3
VEHICLE_WEIGHT = 1
LOG_FILE       = "traffic_log.csv"
FINE_LOG       = "fine_log.csv"

FINE_TABLE = {
    "Red Light Violation": 1000,
    "Overspeed":           2000,
    "Helmet Violation":     500,
    "Mobile Usage":         500,
    "Wrong Side":          1500,
}

CCTV_URLS = ["", "", "", ""]

# ══════════════════════════════════════════════════════════════
#  LOCATION + JUNCTION DATA
# ══════════════════════════════════════════════════════════════

LOCATIONS = {
    "Tamil Nadu": {
        "Chennai": {
            "Tambaram":   ["Indira Nagar Junction", "Camp Road Junction", "Mupet Junction"],
            "Velachery":  ["Velachery Main Jn", "Taramani Junction"],
            "Anna Nagar": ["Anna Nagar Roundtana", "Thirumangalam Junction"],
            "T Nagar":    ["T Nagar Bus Terminus Jn", "Pondy Bazaar Junction"],
        },
        "Coimbatore": {
            "RS Puram":  ["Gandhipuram Junction", "Town Hall Junction"],
            "Peelamedu": ["Peelamedu Junction",   "Avinashi Road Junction"],
        },
        "Madurai": {
            "Anna Nagar": ["Mattuthavani Junction", "Bypass Road Junction"],
        },
    }
}

JUNCTION_LANES = {
    "Indira Nagar Junction": 4,   "Camp Road Junction": 3,
    "Mupet Junction": 4,          "Velachery Main Jn": 4,
    "Taramani Junction": 3,       "Gandhipuram Junction": 4,
    "Town Hall Junction": 3,      "Anna Nagar Roundtana": 4,
    "Thirumangalam Junction": 4,  "T Nagar Bus Terminus Jn": 4,
    "Pondy Bazaar Junction": 3,   "Peelamedu Junction": 4,
    "Avinashi Road Junction": 3,  "Mattuthavani Junction": 4,
    "Bypass Road Junction": 3,
}

LOCATION_COORDS = {
    "Chennai":    (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai":    (9.9252,  78.1198),
}

# ── Junction Risk & Infrastructure ────────────────────────────
JUNCTION_RISK = {
    "Indira Nagar Junction":   {"risk":"HIGH",   "school":True,  "hospital":True,  "crossings":4, "daily":12000},
    "Camp Road Junction":      {"risk":"MEDIUM", "school":False, "hospital":False, "crossings":3, "daily":8000},
    "Mupet Junction":          {"risk":"HIGH",   "school":True,  "hospital":False, "crossings":4, "daily":15000},
    "Velachery Main Jn":       {"risk":"MEDIUM", "school":False, "hospital":True,  "crossings":4, "daily":18000},
    "Taramani Junction":       {"risk":"LOW",    "school":False, "hospital":False, "crossings":2, "daily":6000},
    "Gandhipuram Junction":    {"risk":"HIGH",   "school":False, "hospital":True,  "crossings":6, "daily":22000},
    "Town Hall Junction":      {"risk":"MEDIUM", "school":False, "hospital":False, "crossings":3, "daily":9000},
    "Anna Nagar Roundtana":    {"risk":"HIGH",   "school":True,  "hospital":False, "crossings":6, "daily":20000},
    "Thirumangalam Junction":  {"risk":"MEDIUM", "school":False, "hospital":True,  "crossings":4, "daily":14000},
    "T Nagar Bus Terminus Jn": {"risk":"HIGH",   "school":False, "hospital":False, "crossings":5, "daily":25000},
    "Pondy Bazaar Junction":   {"risk":"HIGH",   "school":False, "hospital":False, "crossings":4, "daily":19000},
    "Peelamedu Junction":      {"risk":"MEDIUM", "school":True,  "hospital":True,  "crossings":4, "daily":11000},
    "Avinashi Road Junction":  {"risk":"LOW",    "school":False, "hospital":False, "crossings":2, "daily":7000},
    "Mattuthavani Junction":   {"risk":"HIGH",   "school":False, "hospital":True,  "crossings":4, "daily":16000},
    "Bypass Road Junction":    {"risk":"LOW",    "school":False, "hospital":False, "crossings":2, "daily":5000},
}

# ══════════════════════════════════════════════════════════════
#  BUS ROUTE DATABASE
# ══════════════════════════════════════════════════════════════

BUS_ROUTES = {
    "Chennai": {
        "Indira Nagar Junction": [
            {"route":"21C",  "from":"Tambaram",   "to":"Broadway",        "freq":8,  "peaks":["07:00-09:00","17:00-20:00"], "type":"AC"},
            {"route":"47",   "from":"Tambaram",   "to":"Guindy",          "freq":12, "peaks":["06:00-09:00","17:00-21:00"], "type":"Ordinary"},
            {"route":"500",  "from":"Tambaram",   "to":"Koyambedu",       "freq":15, "peaks":["07:00-10:00","17:00-20:00"], "type":"Mini"},
            {"route":"63A",  "from":"Velachery",  "to":"Thiruvanmiyur",   "freq":10, "peaks":["07:00-09:30","18:00-20:30"], "type":"Ordinary"},
            {"route":"M77",  "from":"Tambaram",   "to":"Central Station", "freq":20, "peaks":["06:30-09:00","17:00-20:00"], "type":"MTC Express"},
        ],
        "Camp Road Junction": [
            {"route":"23",   "from":"Tambaram",   "to":"Perambur",        "freq":10, "peaks":["07:00-09:00","17:00-20:00"], "type":"Ordinary"},
            {"route":"45D",  "from":"Chromepet",  "to":"Broadway",        "freq":15, "peaks":["07:30-09:30","17:30-20:30"], "type":"Ordinary"},
        ],
        "Mupet Junction": [
            {"route":"51",   "from":"Tambaram",   "to":"T Nagar",         "freq":6,  "peaks":["06:30-09:30","17:00-21:00"], "type":"Ordinary"},
            {"route":"180",  "from":"Airport",    "to":"Broadway",        "freq":25, "peaks":["05:00-08:00","20:00-23:00"], "type":"AC"},
        ],
        "Velachery Main Jn": [
            {"route":"V1",   "from":"Velachery",  "to":"Airport",         "freq":15, "peaks":["05:00-08:00","18:00-22:00"], "type":"AC"},
            {"route":"V2",   "from":"Velachery",  "to":"Tambaram",        "freq":10, "peaks":["06:30-09:30","17:00-20:30"], "type":"Ordinary"},
            {"route":"M36",  "from":"Velachery",  "to":"Central",         "freq":20, "peaks":["07:00-09:00","17:30-20:30"], "type":"MTC Express"},
        ],
        "Taramani Junction": [
            {"route":"T3",   "from":"Taramani",   "to":"Broadway",        "freq":12, "peaks":["07:00-09:00","17:00-20:00"], "type":"Ordinary"},
        ],
        "Anna Nagar Roundtana": [
            {"route":"A3",   "from":"Anna Nagar", "to":"Koyambedu",       "freq":8,  "peaks":["06:30-09:30","17:00-20:30"], "type":"Ordinary"},
            {"route":"A9",   "from":"Anna Nagar", "to":"Egmore",          "freq":10, "peaks":["07:00-09:00","17:30-20:00"], "type":"Ordinary"},
            {"route":"101",  "from":"Anna Nagar", "to":"Central Station", "freq":15, "peaks":["07:00-09:30","17:00-20:30"], "type":"AC"},
        ],
        "Thirumangalam Junction": [
            {"route":"T9",   "from":"Thirumangalam","to":"Broadway",       "freq":10, "peaks":["07:00-09:00","17:00-20:30"], "type":"Ordinary"},
        ],
        "T Nagar Bus Terminus Jn": [
            {"route":"5C",   "from":"T Nagar",    "to":"Kasimedu",        "freq":5,  "peaks":["06:00-09:30","17:00-21:00"], "type":"Ordinary"},
            {"route":"11",   "from":"T Nagar",    "to":"Redhills",        "freq":8,  "peaks":["06:30-09:30","16:30-20:30"], "type":"Ordinary"},
            {"route":"44",   "from":"T Nagar",    "to":"Perambur",        "freq":10, "peaks":["07:00-09:00","17:30-20:30"], "type":"Ordinary"},
            {"route":"29C",  "from":"T Nagar",    "to":"Tambaram",        "freq":12, "peaks":["07:00-09:30","17:00-20:30"], "type":"Ordinary"},
        ],
        "Pondy Bazaar Junction": [
            {"route":"PB1",  "from":"Pondy Bazaar","to":"Broadway",       "freq":7,  "peaks":["07:00-09:00","17:00-20:30"], "type":"Ordinary"},
            {"route":"PB5",  "from":"Pondy Bazaar","to":"Tambaram",       "freq":12, "peaks":["07:30-09:30","17:00-21:00"], "type":"Ordinary"},
        ],
    },
    "Coimbatore": {
        "Gandhipuram Junction": [
            {"route":"G1",   "from":"Gandhipuram","to":"Ukkadam",         "freq":6,  "peaks":["07:00-09:30","17:00-20:30"], "type":"Ordinary"},
            {"route":"G15",  "from":"Gandhipuram","to":"Peelamedu",       "freq":10, "peaks":["07:30-09:30","17:00-20:00"], "type":"AC"},
            {"route":"G9",   "from":"Gandhipuram","to":"Airport",         "freq":20, "peaks":["05:00-08:00","18:00-22:00"], "type":"AC"},
        ],
        "Town Hall Junction": [
            {"route":"TH1",  "from":"Town Hall",  "to":"RS Puram",        "freq":8,  "peaks":["07:00-09:30","17:00-20:30"], "type":"Ordinary"},
        ],
        "Peelamedu Junction": [
            {"route":"P12",  "from":"Peelamedu",  "to":"Gandhipuram",     "freq":10, "peaks":["06:30-09:30","17:00-20:30"], "type":"Ordinary"},
        ],
        "Avinashi Road Junction": [
            {"route":"AR3",  "from":"Avinashi Rd","to":"Gandhipuram",     "freq":15, "peaks":["07:00-09:00","17:30-20:00"], "type":"Ordinary"},
        ],
    },
    "Madurai": {
        "Mattuthavani Junction": [
            {"route":"M1",   "from":"Mattuthavani","to":"Bypass",         "freq":10, "peaks":["07:00-09:30","17:00-20:30"], "type":"Ordinary"},
            {"route":"M5",   "from":"Mattuthavani","to":"Meenakshi Amman","freq":8,  "peaks":["06:30-09:30","17:00-21:00"], "type":"Ordinary"},
        ],
        "Bypass Road Junction": [
            {"route":"B3",   "from":"Bypass",     "to":"Mattuthavani",    "freq":12, "peaks":["07:00-09:00","17:00-20:30"], "type":"Ordinary"},
        ],
    },
}

# ── Vehicle classes ────────────────────────────────────────────
# ── 11 trained classes — must match data.yaml ─────────────────
ALL_VEHICLE_CLASSES = {
    "ambulance", "auto", "bicycle", "bus", "car",
    "concrete_mixer", "tempo", "tractor",
    "truck_tanker", "two_wheelers", "vehicle_truck"
}
BUS_CLASSES   = {"bus"}
MOTO_CLASSES  = {"bicycle", "two_wheelers", "auto", "tempo"}
CAR_CLASSES   = {"car"}
LARGE_CLASSES = {"vehicle_truck", "truck_tanker", "bus", "concrete_mixer", "tractor"}

# ══════════════════════════════════════════════════════════════
#  SESSION INIT
# ══════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "video_started":       False,
        "video_paths":         [],
        "video_bytes":         {},
        "alert_active":        False,
        "alert_time":          0.0,
        "current_alert":       "",
        "violations":          [],
        "traffic_history":     [],
        "lane_counts":         [0,0,0,0],
        "bus_count":           0,
        "controller":          None,
        "last_alert_time":     0.0,
        "last_siren_check":    time.time(),   # ← start NOW so siren doesn't fire on first load
        "prev_time":           time.time(),
        "frame_skip":          0,
        "frame_count":         0,
        "prev_positions":      {},
        "detected_plates":     [],
        "emergency_active":    False,
        "ambulance_active":    False,
        "ambulance_lane":      0,
        "siren_active":        False,
        "total_fines":         0,
        "last_conf":           0.0,
        "last_vtype":          {},
        "incident_log":        [],
        "ped_events":          [],
        "ped_crosswalk_count": 0,
        "ped_danger_count":    0,
        "bus_log":             [],
        "signal_log":          [],
        "_caps":               [],    # persistent VideoCapture objects
        "_cap_paths":          [],    # paths the caps were opened for
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ══════════════════════════════════════════════════════════════
#  HELPERS
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
# ══════════════════════════════════════════════════════════════

class TrafficSignalController:
    def __init__(self, num_lanes=4):
        self.num_lanes   = num_lanes
        self.current     = 0
        self.phase       = "GREEN"
        self.t_start     = time.time()
        self.green_times = [BASE_GREEN] * num_lanes

    def update(self, lane_counts, ambulance_lane=-1, siren=False):
        now     = time.time()
        elapsed = now - self.t_start

        # ── AMBULANCE PRIORITY: force green on ambulance lane ──
        if ambulance_lane >= 0 or siren:
            target = ambulance_lane if ambulance_lane >= 0 else self.current
            if self.current != target:
                # Yellow transition before switching
                if self.phase != "YELLOW":
                    self.phase   = "YELLOW"
                    self.t_start = now
                    return
                elif elapsed >= YELLOW_TIME:
                    self.current = target
                    self.phase   = "GREEN"
                    self.t_start = now
            else:
                if self.phase != "GREEN":
                    self.phase   = "GREEN"
                    self.t_start = now
            return

        # ── NORMAL ADAPTIVE SIGNAL ─────────────────────────────
        if not lane_counts or max(lane_counts) == 0:
            cur_green = BASE_GREEN
        else:
            for i, cnt in enumerate(lane_counts):
                self.green_times[i] = min(BASE_GREEN + cnt * 1.8, MAX_GREEN)
            cur_green = self.green_times[self.current]

        if self.phase == "GREEN" and elapsed >= cur_green:
            self.phase   = "YELLOW"
            self.t_start = now
        elif self.phase == "YELLOW" and elapsed >= YELLOW_TIME:
            self.phase   = "GREEN"
            if lane_counts and max(lane_counts) > 0:
                self.current = lane_counts.index(max(lane_counts))
            else:
                self.current = (self.current + 1) % self.num_lanes
            self.t_start = now

    def get_light(self, lane):
        if lane == self.current:
            return self.phase
        return "RED"

    def time_remaining(self, lane):
        elapsed = time.time() - self.t_start
        light   = self.get_light(lane)
        if light == "GREEN":
            return max(0, int(self.green_times[self.current] - elapsed))
        elif light == "YELLOW":
            return max(0, int(YELLOW_TIME - elapsed))
        return "—"

# ══════════════════════════════════════════════════════════════
#  ALERT SYSTEM
# ══════════════════════════════════════════════════════════════

def _beep():
    if WINSOUND_OK:
        for _ in range(4):
            winsound.Beep(1500, 250)

NTFY_TOPIC   = "smart_traffic_alerts_v4"   # change to your ntfy topic name
NTFY_SERVER  = "https://ntfy.sh"            # or your self-hosted server

def send_push(msg, title="Traffic Alert", priority="default", tags=None, event_type="alert"):
    """Send rich ntfy push notification. All strings kept ASCII-safe for headers."""
    try:
        topic = st.session_state.get("ntfy_topic_val", NTFY_TOPIC)

        # HTTP headers must be ASCII — use plain text titles, put emoji in message body
        if event_type == "ambulance":
            priority = "urgent"
            tags     = "rotating_light,ambulance,red_circle"
            title    = "AMBULANCE PRIORITY - Clear the lane NOW"
        elif event_type == "siren":
            priority = "urgent"
            tags     = "rotating_light,loud_sound,red_circle"
            title    = "AMBULANCE SIREN DETECTED"
        elif event_type == "redlight":
            priority = "high"
            tags     = "red_circle,rotating_light,oncoming_police_car"
            title    = "Red Light Violation"
        elif event_type == "overspeed":
            priority = "high"
            tags     = "racing_car,rotating_light,oncoming_police_car"
            title    = "Overspeed Violation"
        elif event_type == "helmet":
            priority = "default"
            tags     = "helmet,motorbike,warning"
            title    = "Helmet Violation"
        elif event_type == "pedestrian":
            priority = "high"
            tags     = "walking,warning,red_circle"
            title    = "Pedestrian in Danger"
        elif event_type == "congestion":
            priority = "default"
            tags     = "traffic_light,warning"
            title    = "High Congestion Alert"
        else:
            tags  = tags if isinstance(tags, str) else "warning"
            title = title if title else "Traffic Alert"

        # Encode headers as ASCII (required by HTTP spec)
        # Strip any non-ASCII from title just in case
        safe_title = title.encode("ascii", errors="ignore").decode("ascii")
        safe_tags  = tags if isinstance(tags, str) else "warning"

        requests.post(
            "https://ntfy.sh/" + topic,
            data=msg.encode("utf-8"),
            headers={
                "Title":        safe_title,
                "Priority":     priority,
                "Tags":         safe_tags,
                "Content-Type": "text/plain; charset=utf-8",
            },
            timeout=4
        )
    except Exception:
        pass

def trigger_alert(event, plate="", lane=-1):
    now = time.time()
    if now - st.session_state.last_alert_time < 4:
        return
    st.session_state.last_alert_time = now
    st.session_state.alert_active    = True
    st.session_state.alert_time      = now
    st.session_state.current_alert   = event

    is_amb = any(k in event for k in ["Ambulance","Siren","🚑"])
    is_ped = any(k in event for k in ["Pedestrian","Crosswalk","🚶"])
    fine   = 0 if (is_amb or is_ped) else FINE_TABLE.get(event, 500)
    if not (is_amb or is_ped):
        st.session_state.total_fines += fine

    status = ("EXEMPT — Emergency Vehicle" if is_amb
              else "ALERT — Pedestrian Safety" if is_ped
              else "Pending")
    st.session_state.violations.append({
        "Time":     datetime.now().strftime("%H:%M:%S"),
        "Event":    event,
        "Plate":    plate if plate else "—",
        "Lane":     f"Lane {lane+1}" if lane >= 0 else "—",
        "Fine(Rs)": fine,
        "Status":   status,
    })
    try:
        with open(FINE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, plate, fine])
    except Exception:
        pass
    threading.Thread(target=_beep, daemon=True).start()

    # ── Determine event type for rich ntfy notification ─────────
    if "Ambulance" in event:
        etype = "ambulance"
    elif "Siren" in event:
        etype = "siren"
    elif "Red Light" in event:
        etype = "redlight"
    elif "Overspeed" in event or "speed" in event.lower():
        etype = "overspeed"
    elif "Helmet" in event:
        etype = "helmet"
    elif "Pedestrian" in event or "Crosswalk" in event:
        etype = "pedestrian"
    elif "Congestion" in event:
        etype = "congestion"
    else:
        etype = "alert"

    lane_str  = f"Lane {lane+1}" if lane >= 0 else "Unknown Lane"
    plate_str = f" | Plate: {plate}" if plate else ""
    fine_str  = f" | Fine: Rs {fine}" if fine > 0 else " | EXEMPT"
    push_msg  = f"{event}{plate_str} | {lane_str}{fine_str} | {datetime.now().strftime('%H:%M:%S')}"
    threading.Thread(target=send_push, args=(push_msg,), kwargs={"event_type": etype}, daemon=True).start()

def show_alert_banner():
    if not st.session_state.alert_active:
        return
    if time.time() - st.session_state.alert_time > 12:
        st.session_state.alert_active = False
        return
    msg    = st.session_state.current_alert
    icon_map = {
        "Ambulance":"🚑","Siren":"🚑","Red Light":"🚨","Overspeed":"🚗",
        "Helmet":"🪖","Mobile":"📵","Accident":"💥","Pedestrian":"🚶","Crosswalk":"🚶",
    }
    icon   = next((v for k, v in icon_map.items() if k in msg), "⚠️")
    is_amb = "Ambulance" in msg or "Siren" in msg
    is_ped = "Pedestrian" in msg or "Crosswalk" in msg
    is_spd = "Overspeed" in msg or "speed" in msg.lower()
    is_rl  = "Red Light" in msg
    is_hlm = "Helmet" in msg
    is_mob = "Mobile" in msg

    if is_amb:
        bg  = "linear-gradient(90deg,#000055,#0000cc)"
        bdr = "#4466ff"
        anim= "animation:amb_flash 0.5s infinite;"
    elif is_ped:
        bg  = "linear-gradient(90deg,#003300,#006600)"
        bdr = "#00ff88"
        anim= ""
    elif is_rl:
        bg  = "linear-gradient(90deg,#550000,#cc0000)"
        bdr = "#ff0000"
        anim= "animation:pulse 0.4s infinite;"
    elif is_spd:
        bg  = "linear-gradient(90deg,#332200,#cc6600)"
        bdr = "#ff8800"
        anim= "animation:pulse 0.6s infinite;"
    elif is_hlm:
        bg  = "linear-gradient(90deg,#223300,#557700)"
        bdr = "#aaff00"
        anim= ""
    elif is_mob:
        bg  = "linear-gradient(90deg,#220033,#660099)"
        bdr = "#cc00ff"
        anim= ""
    else:
        bg  = "linear-gradient(90deg,#8B0000,#cc0000)"
        bdr = "#ff4444"
        anim= ""

    st.markdown(f"""
    <div style="background:{bg};padding:18px 28px;border-radius:12px;text-align:center;
                font-size:24px;font-weight:700;color:#fff;{anim}
                box-shadow:0 0 40px {bdr}88;margin-bottom:14px;border:2px solid {bdr};
                letter-spacing:1px;">
      {icon}&nbsp;&nbsp;{msg}&nbsp;&nbsp;{icon}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  WEATHER
# ══════════════════════════════════════════════════════════════

def get_weather(city):
    lat, lon = LOCATION_COORDS.get(city, (13.08, 80.27))
    try:
        r  = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true",
            timeout=4)
        cw   = r.json().get("current_weather", {})
        temp = cw.get("temperature", "?")
        wc   = cw.get("weathercode", 0)
        icon = "☀️" if wc < 3 else "🌤" if wc < 50 else "🌧" if wc < 70 else "⛈"
        return f"{icon} {temp} °C"
    except Exception:
        return "🌡 N/A"

# ══════════════════════════════════════════════════════════════
#  SIREN DETECTION
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_siren_model():
    try:
        with open("siren_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def detect_siren():
    if not AUDIO_AVAILABLE or not LIBROSA_OK:
        return False
    data = load_siren_model()
    if data is None:
        return False
    try:
        SR, N_MFCC, DURATION = data.get("sr",22050), data.get("n_mfcc",40), data.get("duration",3.0)
        audio = sd.rec(int(SR*DURATION), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        y      = audio.flatten()
        mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=y, sr=SR)
        zcr    = librosa.feature.zero_crossing_rate(y)
        rmse   = librosa.feature.rms(y=y)
        feat   = np.concatenate([np.mean(mfcc,axis=1), np.std(mfcc,axis=1),
                                  np.mean(chroma,axis=1), [np.mean(zcr), np.mean(rmse)]]).reshape(1,-1)
        pred  = data["model"].predict(feat)[0]
        label = str(data["encoder"].inverse_transform([pred])[0])
        return label == "ambulance"
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════
#  VISION HEURISTICS
# ══════════════════════════════════════════════════════════════

def is_ambulance_roi(roi):
    if roi is None or roi.size < 400:
        return False
    try:
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        wht  = cv2.inRange(hsv, (0,0,200), (180,40,255))
        r1   = cv2.inRange(hsv, (0,120,70),  (10,255,255))
        r2   = cv2.inRange(hsv, (170,120,70),(180,255,255))
        red  = cv2.bitwise_or(r1, r2)
        return (np.count_nonzero(wht)/roi.size > 0.35 or np.count_nonzero(red)/roi.size > 0.04)
    except Exception:
        return False

def check_helmet(head_roi):
    if head_roi is None or head_roi.size < 100:
        return True
    try:
        hsv  = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, (0,30,60), (20,150,255))
        return np.count_nonzero(skin)/max(head_roi.size,1) < 0.55
    except Exception:
        return True

# ══════════════════════════════════════════════════════════════
#  PEDESTRIAN SAFETY
# ══════════════════════════════════════════════════════════════

def detect_pedestrian_danger(frame_infer, signal_light, lane_idx, disp_frame):
    """Detects pedestrian presence in crosswalk zone via skin-tone proxy."""
    h, w   = disp_frame.shape[:2]
    cw_y1  = int(h * 0.70)

    # Draw zebra crossing stripes
    stripe_h = (h - cw_y1) // 5
    for s in range(5):
        y     = cw_y1 + s * stripe_h
        color = (180,180,180) if s % 2 == 0 else (80,80,80)
        cv2.rectangle(disp_frame, (0,y), (w, y+stripe_h), color, 1)

    # Skin-tone blob in lower 30% of inference frame
    ih, iw  = frame_infer.shape[:2]
    roi     = frame_infer[int(ih*0.70):, :]
    ped_det = False
    try:
        hsv      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin     = cv2.inRange(hsv, (0,25,50), (22,200,255))
        ped_det  = np.count_nonzero(skin) > getattr(st.session_state, "ped_conf_val", 400)
    except Exception:
        pass

    danger = ped_det and signal_light == "GREEN"

    if ped_det:
        st.session_state.ped_crosswalk_count += 1
        color = (0,0,255) if danger else (0,255,100)
        cv2.rectangle(disp_frame, (0,cw_y1), (w,h), color, 3)
        cv2.putText(disp_frame, "CROSSWALK" + (" DANGER!" if danger else ""),
                    (5, cw_y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if danger:
        st.session_state.ped_danger_count += 1
        trigger_alert("🚶 Pedestrian in Crosswalk — Vehicles STOP!", lane=lane_idx)
        st.session_state.ped_events.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "lane": lane_idx+1,
            "type": "Vehicle vs Pedestrian Danger",
        })
    return ped_det, danger

# ══════════════════════════════════════════════════════════════
#  BUS SCHEDULE TRACKER  (Modern Metro Board)
# ══════════════════════════════════════════════════════════════

def get_buses_arriving_at_junction(junction, city, start_str, end_str):
    """Return ONLY buses that are arriving/passing THROUGH this junction (inbound to its area)."""
    routes = BUS_ROUTES.get(city, {}).get(junction, [])
    if not routes:
        return []
    fmt = "%H:%M"
    try:
        t_s = datetime.strptime(start_str, fmt)
        t_e = datetime.strptime(end_str,   fmt)
        if t_e <= t_s:
            t_e += timedelta(hours=24)
    except Exception:
        return []

    # Derive locality keywords from junction name for matching
    junc_area = (junction
                 .replace(" Junction","").replace(" Jn","")
                 .replace(" Terminus","").replace(" Roundtana","")
                 .strip().lower())
    # Also pull area name keywords
    area_kw = [w.lower() for w in junc_area.split() if len(w) > 3]

    def _is_arriving(route):
        """True if this bus ARRIVES at (or passes through) this junction's locality."""
        dest = route["to"].lower()
        src  = route["from"].lower()
        # If destination matches junction area → definitely inbound
        if any(kw in dest for kw in area_kw):
            return True, "⬇️ Arriving"
        # If source matches junction area → outbound (leaving), skip
        if any(kw in src for kw in area_kw):
            return False, "⬆️ Departing"
        # Route is registered AT this junction but neither from nor to matches —
        # treat as passing through (show it)
        return True, "↔️ Via Junction"

    results = []
    for route in routes:
        is_inbound, direction = _is_arriving(route)
        if not is_inbound:
            continue   # Skip buses LEAVING from this junction

        freq       = route["freq"]
        cursor     = datetime.strptime("05:00", fmt)
        end_of_day = datetime.strptime("23:00", fmt)
        while cursor <= end_of_day:
            if t_s <= cursor <= t_e:
                now       = datetime.now()
                bus_dt    = now.replace(hour=cursor.hour, minute=cursor.minute, second=0, microsecond=0)
                mins_away = int((bus_dt - now).total_seconds() / 60)
                if mins_away < 0:
                    mins_away += 1440
                is_peak = False
                for ph in route.get("peaks", []):
                    try:
                        ph_s, ph_e = ph.split("-")
                        if datetime.strptime(ph_s, fmt) <= cursor <= datetime.strptime(ph_e, fmt):
                            is_peak = True; break
                    except Exception:
                        pass
                results.append({
                    "Route":      route["route"],
                    "From":       route["from"],
                    "To":         route["to"],
                    "Direction":  direction,
                    "Type":       route["type"],
                    "Time":       cursor.strftime("%H:%M"),
                    "Mins Away":  max(0, mins_away),
                    "Peak Hour":  "🔴 Peak" if is_peak else "🟢 Normal",
                    "Freq (min)": freq,
                })
            cursor += timedelta(minutes=freq)
    results.sort(key=lambda x: x["Mins Away"])
    return results

# ══════════════════════════════════════════════════════════════
#  JUNCTION INTELLIGENCE CARD
# ══════════════════════════════════════════════════════════════

def render_junction_intelligence(junction, city):
    info = JUNCTION_RISK.get(junction, {"risk":"MEDIUM","school":False,"hospital":False,"crossings":2,"daily":5000})
    rc   = {"HIGH":"#ff4444","MEDIUM":"#ffcc00","LOW":"#00ee88"}.get(info["risk"],"#aaa")
    ri   = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(info["risk"],"⚪")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1f33,#0a1520);border:1px solid {rc}44;
                border-radius:14px;padding:18px 22px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
        <div>
          <div style="font-family:'Orbitron',monospace;font-size:12px;color:#00ffcc;letter-spacing:2px;">
            JUNCTION INTELLIGENCE
          </div>
          <div style="font-size:20px;font-weight:700;color:#ffffff;margin-top:4px;">{junction}</div>
          <div style="font-size:12px;color:#4a6a8a;">{city}</div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <div style="background:#111827;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px;">
            <div style="font-size:20px;">{ri}</div>
            <div style="font-size:9px;color:#4a6a8a;">RISK</div>
            <div style="font-size:12px;font-weight:700;color:{rc};">{info['risk']}</div>
          </div>
          <div style="background:#111827;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px;">
            <div style="font-size:20px;">🚶</div>
            <div style="font-size:9px;color:#4a6a8a;">CROSSINGS</div>
            <div style="font-size:12px;font-weight:700;color:#88aaff;">{info['crossings']}</div>
          </div>
          <div style="background:#111827;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px;">
            <div style="font-size:20px;">🚗</div>
            <div style="font-size:9px;color:#4a6a8a;">DAILY AVG</div>
            <div style="font-size:12px;font-weight:700;color:#00ffcc;">{info['daily']:,}</div>
          </div>
          <div style="background:#111827;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px;">
            <div style="font-size:20px;">{'🏫' if info['school'] else '🏢'}</div>
            <div style="font-size:9px;color:#4a6a8a;">SCHOOL ZONE</div>
            <div style="font-size:12px;font-weight:700;color:{'#ff8844' if info['school'] else '#4a6a8a'};">
              {'YES ⚠️' if info['school'] else 'NO'}
            </div>
          </div>
          <div style="background:#111827;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px;">
            <div style="font-size:20px;">{'🏥' if info['hospital'] else '🏪'}</div>
            <div style="font-size:9px;color:#4a6a8a;">HOSPITAL</div>
            <div style="font-size:12px;font-weight:700;color:{'#ff5555' if info['hospital'] else '#4a6a8a'};">
              {'YES 🚑' if info['hospital'] else 'NO'}
            </div>
          </div>
        </div>
      </div>
      {'<div style="background:#ff440011;border:1px solid #ff4444;border-radius:6px;padding:6px 12px;margin-top:10px;font-size:12px;color:#ffaaaa;">⚠️ SCHOOL ZONE ACTIVE — Reduced speed limits enforced. Extended pedestrian signal time.</div>' if info['school'] else ''}
      {'<div style="background:#ff000011;border:1px solid #ff5555;border-radius:6px;padding:6px 12px;margin-top:6px;font-size:12px;color:#ffaaaa;">🚑 HOSPITAL ZONE — Ambulance corridor auto-cleared. Zero tolerance for violations.</div>' if info['hospital'] else ''}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    candidates = [
        MODEL_NAME,
        r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\runs\detect\runs\smart_traffic_fixed\weights\best.pt",
        r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\runs\detect\runs\smart_traffic_fixed\weights\last.pt",
        r"F:\major_project\YOLOv8-DeepSORT-Streamlit\SmartTraffic\yolov8n.pt",
        "yolov8n.pt",
    ]
    for p in candidates:
        if Path(p).exists():
            return YOLO(p)
    st.warning("⚠️ Custom weights not found — falling back to pretrained yolov8n.pt")
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.65}}
@keyframes amb_flash{0%,100%{background:linear-gradient(90deg,#000055,#0000cc)}
                     50%{background:linear-gradient(90deg,#0000cc,#000055)}}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background-color:#080c14;}
section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e2d40;}
.sidebar-logo{font-family:'Orbitron',monospace;font-size:16px;font-weight:700;color:#00ffcc;
              text-align:center;padding:12px 0 4px;letter-spacing:2px;}
.sidebar-sub{color:#3a5a7a;font-size:11px;text-align:center;margin-bottom:12px;}
.sb-section{background:#111827;border-radius:6px;padding:8px 12px;margin:8px 0;
            border-left:3px solid #00ffcc;font-size:11px;color:#557;
            font-weight:700;letter-spacing:1px;text-transform:uppercase;}
.card{background:linear-gradient(135deg,#0f1923,#1a2535);border:1px solid #1e3a5f;
      padding:16px 12px;border-radius:12px;text-align:center;margin-bottom:8px;
      box-shadow:0 4px 20px rgba(0,255,204,0.06);}
.card-val{font-family:'Orbitron',monospace;font-size:28px;color:#00ffcc;font-weight:700;}
.card-lbl{color:#4a6a8a;font-size:11px;margin-top:4px;text-transform:uppercase;letter-spacing:1px;}
.alert-card{background:linear-gradient(90deg,#4a0000,#800000);border:1px solid #ff3333;
            padding:12px 18px;border-radius:8px;color:white;font-weight:700;font-size:15px;
            box-shadow:0 0 18px #ff000055;margin:5px 0;}
.ambu-card{background:linear-gradient(90deg,#000033,#000099);border:1px solid #4466ff;
           padding:12px 18px;border-radius:8px;color:white;font-weight:700;font-size:15px;
           box-shadow:0 0 18px #0000ff55;margin:5px 0;animation:amb_flash 0.6s infinite;}
.page-title{font-family:'Orbitron',monospace;font-size:26px;color:#00ffcc;
            text-align:center;letter-spacing:4px;padding:10px 0 2px;
            text-shadow:0 0 20px #00ffcc55;}
.page-sub{color:#3a5a7a;font-size:13px;text-align:center;margin-bottom:14px;}
.sig-box{background:#0a0f1a;border:1px solid #1e2d40;border-radius:12px;
         padding:14px 18px;text-align:center;min-width:100px;}
.sig-lane{font-family:'Orbitron',monospace;font-size:12px;color:#00ffcc;}
.sig-count{color:#4a6a8a;font-size:11px;margin:3px 0;}
.divider{border-top:1px solid #1a2535;margin:16px 0;}
.bus-row{background:#0a0f1a;border:1px solid #1e3a5f;border-radius:8px;
         padding:10px 14px;margin:4px 0;display:flex;align-items:center;gap:12px;}
.amb-banner{background:linear-gradient(90deg,#000055,#0000aa55);border:2px solid #4466ff;
            border-radius:10px;padding:18px;text-align:center;animation:amb_flash 0.5s infinite;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD RESOURCES
# ══════════════════════════════════════════════════════════════

model  = load_model()
reader = load_ocr()
MODEL_CLASSES = set(model.names.values()) if model.names else set()

if st.session_state.controller is None:
    st.session_state.controller = TrafficSignalController()
controller = st.session_state.controller

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    # ══ HEADER ═════════════════════════════════════════════════
    st.markdown("""
    <div style="text-align:center;padding:10px 0 4px;">
      <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:700;
                  color:#00ffcc;letter-spacing:3px;text-shadow:0 0 12px #00ffcc66;">
        🚦 TRAFFIC AI v4.0
      </div>
      <div style="font-size:10px;color:#2a4a6a;letter-spacing:2px;margin-top:2px;">
        SMART CITY CONTROL SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══ COMMAND CENTER PANEL ═══════════════════════════════════
    now_dt   = datetime.now()
    hour_now = now_dt.hour
    mins_now = now_dt.minute

    shift_name  = ("MORNING SHIFT" if 6<=hour_now<14 else
                   "EVENING SHIFT" if 14<=hour_now<22 else "NIGHT SHIFT")
    shift_icon  = "🌅" if 6<=hour_now<14 else "🌆" if 14<=hour_now<22 else "🌙"
    shift_range = ("06:00 – 14:00" if 6<=hour_now<14 else
                   "14:00 – 22:00" if 14<=hour_now<22 else "22:00 – 06:00")
    is_peak     = (7<=hour_now<9) or (17<=hour_now<20)
    is_school   = (8<=hour_now<9) or (15<=hour_now<16)
    peak_label  = "🔴 PEAK TRAFFIC" if is_peak else "🟢 NORMAL FLOW"
    peak_clr    = "#ff4444" if is_peak else "#00ee88"
    peak_bg     = "#2a0000" if is_peak else "#002a15"

    # Operator duty timer — minutes into shift
    shift_start = (6 if 6<=hour_now<14 else 14 if 14<=hour_now<22 else 22)
    duty_mins   = (hour_now - shift_start) * 60 + mins_now
    if duty_mins < 0: duty_mins += 1440

    # Pre-compute ALL values so the markdown string has NO single quotes inside {}
    time_str  = now_dt.strftime("%H:%M")
    date_str  = now_dt.strftime("%a %d %b %Y")
    school_badge_html = (
        '<div style="background:#2a1000;border:1px solid #ff8844;border-radius:5px;'
        'padding:3px 8px;font-size:9px;font-weight:700;color:#ff8844;">&#127867; SCHOOL HOURS</div>'
        if is_school else ""
    )

    cmd_html = (
        '<div style="background:linear-gradient(160deg,#040d1a,#071829);'
        'border:1px solid #00ffcc44;border-radius:12px;padding:12px 14px;margin:8px 0;">'

        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
        '<div>'
        '<div style="font-size:8px;color:#00ffcc88;letter-spacing:2px;font-weight:700;">&#9889; COMMAND CENTER</div>'
        '<div style="font-size:15px;font-weight:700;color:#ffffff;margin-top:2px;">'
        + shift_icon + " " + shift_name +
        '</div>'
        '<div style="font-size:10px;color:#3a5a7a;margin-top:1px;">' + shift_range + '</div>'
        '</div>'
        '<div style="text-align:right;">'
        '<div style="font-size:22px;font-weight:700;color:#00ffcc;line-height:1;">' + time_str + '</div>'
        '<div style="font-size:9px;color:#3a5a7a;margin-top:2px;">' + date_str + '</div>'
        '</div>'
        '</div>'

        '<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px;">'
        '<div style="background:' + peak_bg + ';border:1px solid ' + peak_clr + ';border-radius:5px;'
        'padding:3px 8px;font-size:9px;font-weight:700;color:' + peak_clr + ';">' + peak_label + '</div>'
        + school_badge_html +
        '<div style="background:#0a1a2a;border:1px solid #1a3a5a;border-radius:5px;'
        'padding:3px 8px;font-size:9px;color:#4a7a9a;">&#9201; On duty: ' + str(duty_mins) + 'm</div>'
        '</div>'

        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;">'
        '<div style="background:#080f1a;border:1px solid #0d2035;border-radius:7px;padding:8px;text-align:center;">'
        '<div style="font-size:18px;">&#128225;</div>'
        '<div style="font-size:8px;color:#2a5a7a;font-weight:700;letter-spacing:1px;margin-top:2px;">NTFY ALERTS</div>'
        '<div style="font-size:9px;color:#00ee88;margin-top:2px;">&#9679; LIVE</div>'
        '</div>'
        '<div style="background:#080f1a;border:1px solid #0d2035;border-radius:7px;padding:8px;text-align:center;">'
        '<div style="font-size:18px;">&#129302;</div>'
        '<div style="font-size:8px;color:#2a5a7a;font-weight:700;letter-spacing:1px;margin-top:2px;">AI DETECTION</div>'
        '<div style="font-size:9px;color:#00ee88;margin-top:2px;">&#9679; ACTIVE</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(cmd_html, unsafe_allow_html=True)

    # ══ ACTIVE DEPLOYMENT ZONE ═════════════════════════════════
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:8px;color:#00ffcc88;
                letter-spacing:2px;padding:10px 2px 4px;">
      📍 ACTIVE DEPLOYMENT ZONE
    </div>
    """, unsafe_allow_html=True)

    state    = st.selectbox("State", list(LOCATIONS.keys()), key="sel_state", label_visibility="collapsed")
    city_opts = list(LOCATIONS[state].keys())
    city_pop  = {"Chennai":"Chennai · 11M pop · TN Capital",
                 "Coimbatore":"Coimbatore · 2.1M pop · Industrial Hub",
                 "Madurai":"Madurai · 1.5M pop · Temple City"}
    city      = st.selectbox("City", city_opts, key="sel_city", label_visibility="collapsed",
                              format_func=lambda c: city_pop.get(c, c))
    area      = st.selectbox("Zone/Area", list(LOCATIONS[state][city].keys()), key="sel_area", label_visibility="collapsed")
    junction  = st.selectbox("Junction", LOCATIONS[state][city][area], key="sel_junc", label_visibility="collapsed")
    LANES     = JUNCTION_LANES.get(junction, 4)

    # ── Junction Intelligence Card ─────────────────────────────
    j_quick  = JUNCTION_RISK.get(junction, {})
    risk_q   = j_quick.get("risk","MEDIUM")
    rclr_q   = {"HIGH":"#ff4444","MEDIUM":"#ffcc00","LOW":"#00ee88"}.get(risk_q,"#aaa")
    rbg_q    = {"HIGH":"#2a0000","MEDIUM":"#2a2000","LOW":"#002a10"}.get(risk_q,"#111")
    daily_q  = j_quick.get("daily",0)
    hrly_q   = int(daily_q/18) if daily_q else 0
    school_q = j_quick.get("school",False)
    hosp_q   = j_quick.get("hospital",False)
    cross_q  = j_quick.get("crossings",0)
    risk_icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(risk_q,"⚪")

    # Build zone pills using double-quotes only (no single quotes in HTML attrs)
    zone_pills = []
    if school_q:
        zone_pills.append(
            '<span style="background:#2a1000;border:1px solid #ff8844;border-radius:4px;'
            'font-size:9px;color:#ff8844;padding:2px 7px;">&#127867; SCHOOL ZONE</span>'
        )
    if hosp_q:
        zone_pills.append(
            '<span style="background:#2a0008;border:1px solid #ff4466;border-radius:4px;'
            'font-size:9px;color:#ff4466;padding:2px 7px;">&#127973; HOSPITAL ZONE</span>'
        )
    if not school_q and not hosp_q:
        zone_pills.append(
            '<span style="background:#002010;border:1px solid #00cc66;border-radius:4px;'
            'font-size:9px;color:#00cc66;padding:2px 7px;">&#9989; STANDARD ZONE</span>'
        )
    pills_html = " ".join(zone_pills)

    # Pre-compute advisory text
    advisory_lines = []
    if school_q: advisory_lines.append("&#9888;&#65039; Reduced speed + extended ped signal active")
    if hosp_q:   advisory_lines.append("&#128657; Keep ambulance corridor fully clear")
    if not advisory_lines: advisory_lines.append("Standard patrol protocol active")
    advisory_text = " &nbsp;|&nbsp; ".join(advisory_lines)

    hrly_str  = "~{:,}".format(hrly_q)

    junc_html = (
        '<div style="background:linear-gradient(160deg,#060e1a,#0a1828);'
        'border-left:3px solid ' + rclr_q + ';border-radius:10px;padding:12px 14px;margin:4px 0 8px;">'

        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">'
        '<div style="flex:1;">'
        '<div style="font-size:13px;font-weight:700;color:#ffffff;line-height:1.2;">' + junction + '</div>'
        '<div style="font-size:10px;color:#3a5a7a;margin-top:2px;">' + area + ' &middot; ' + city + ' &middot; ' + state + '</div>'
        '</div>'
        '<div style="background:' + rbg_q + ';border:1px solid ' + rclr_q + ';border-radius:6px;'
        'padding:4px 9px;text-align:center;min-width:64px;margin-left:8px;">'
        '<div style="font-size:14px;">' + risk_icon + '</div>'
        '<div style="font-size:8px;font-weight:700;color:' + rclr_q + ';letter-spacing:1px;">' + risk_q + ' RISK</div>'
        '</div>'
        '</div>'

        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:8px;">'
        '<div style="background:#060c16;border-radius:7px;padding:7px;text-align:center;border:1px solid #0d2035;">'
        '<div style="font-size:15px;font-weight:700;color:#00ffcc;">' + hrly_str + '</div>'
        '<div style="font-size:8px;color:#2a4a6a;letter-spacing:1px;margin-top:2px;">VEH/HOUR</div>'
        '</div>'
        '<div style="background:#060c16;border-radius:7px;padding:7px;text-align:center;border:1px solid #0d2035;">'
        '<div style="font-size:15px;font-weight:700;color:#88aaff;">' + str(cross_q) + '</div>'
        '<div style="font-size:8px;color:#2a4a6a;letter-spacing:1px;margin-top:2px;">CROSSINGS</div>'
        '</div>'
        '<div style="background:#060c16;border-radius:7px;padding:7px;text-align:center;border:1px solid #0d2035;">'
        '<div style="font-size:15px;font-weight:700;color:#ffcc00;">' + str(LANES) + '</div>'
        '<div style="font-size:8px;color:#2a4a6a;letter-spacing:1px;margin-top:2px;">LANES</div>'
        '</div>'
        '</div>'

        '<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px;">' + pills_html + '</div>'

        '<div style="background:#04080f;border-radius:6px;padding:7px 10px;border-left:2px solid ' + rclr_q + ';">'
        '<div style="font-size:9px;color:' + rclr_q + ';font-weight:700;letter-spacing:1px;margin-bottom:3px;">'
        '&#128110; OFFICER ADVISORY</div>'
        '<div style="font-size:10px;color:#5a7a9a;">' + advisory_text + '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(junc_html, unsafe_allow_html=True)

    j_info = JUNCTION_RISK.get(junction, {})

    lat, lon = LOCATION_COORDS.get(city, (13.08, 80.27))
    components.html(f"""
    <div style="border-radius:8px;overflow:hidden;border:1px solid #1e3a5f;margin-top:6px;">
      <iframe src="https://www.openstreetmap.org/export/embed.html?bbox={lon-0.03}%2C{lat-0.02}%2C{lon+0.03}%2C{lat+0.02}&layer=mapnik&marker={lat}%2C{lon}"
              style="width:100%;height:180px;border:0;" title="Map"></iframe>
    </div>
    <div style="font-size:10px;color:#2a5a7a;text-align:right;margin-top:2px;">
      📡 {junction} · {lat:.4f}°N, {lon:.4f}°E</div>""", height=200)

    # ── ntfy Push Notifications ────────────────────────────────
    st.markdown('<div class="sb-section">📲 ntfy Push Alerts</div>', unsafe_allow_html=True)

    # Store topic in session_state so send_push can read it globally
    if "ntfy_topic_val" not in st.session_state:
        st.session_state["ntfy_topic_val"] = NTFY_TOPIC

    ntfy_topic_input = st.text_input(
        "ntfy Topic", value=st.session_state["ntfy_topic_val"], key="ntfy_topic",
        placeholder="e.g. smart_traffic_alerts_v4",
        help="Your ntfy.sh topic — subscribe to this in the ntfy app"
    )
    if ntfy_topic_input:
        st.session_state["ntfy_topic_val"] = ntfy_topic_input

    active_topic = st.session_state["ntfy_topic_val"]

    st.markdown(f"""
    <div style="background:#04080f;border:1px solid #0d2035;border-radius:7px;
                padding:8px 10px;font-size:10px;color:#3a5a7a;margin-bottom:6px;">
      <div style="color:#00ffcc;font-weight:700;margin-bottom:4px;">📱 Subscribed Topic</div>
      <div style="font-family:'Orbitron',monospace;font-size:12px;color:#ffffff;
                  background:#080e18;border-radius:4px;padding:4px 8px;
                  border:1px solid #00ffcc44;margin-bottom:6px;">{active_topic}</div>
      <div style="color:#2a5a7a;">Alerts sent for:</div>
      <div style="padding-left:6px;margin-top:3px;line-height:1.7;">
        🚑 <span style="color:#5588ff">Ambulance</span> &nbsp;·&nbsp;
        🔊 <span style="color:#5588ff">Siren</span><br>
        🚨 <span style="color:#ff8844">Red Light</span> &nbsp;·&nbsp;
        🏎️ <span style="color:#ff8844">Overspeed</span><br>
        ⚠️ <span style="color:#ffcc44">Helmet</span> &nbsp;·&nbsp;
        🚶 <span style="color:#ffcc44">Pedestrian</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Test notification button
    if st.button("📲 Send Test Notification", use_container_width=True, key="ntfy_test"):
        try:
            resp = requests.post(
                "https://ntfy.sh/" + active_topic,
                data="Traffic AI v4.0 is connected! You will now receive live alerts for violations, ambulance priority, and pedestrian danger.".encode("utf-8"),
                headers={
                    "Title":        "Traffic AI v4.0 - Connection Test OK",
                    "Priority":     "default",
                    "Tags":         "white_check_mark,traffic_light",
                    "Content-Type": "text/plain; charset=utf-8",
                },
                timeout=5
            )
            if resp.status_code == 200:
                st.success("✅ Test notification sent! Check your ntfy app.")
            else:
                st.error(f"❌ ntfy returned {resp.status_code} — check your topic name.")
        except Exception as e:
            st.error(f"❌ Send failed: {e}")

    # ── Camera Source ──────────────────────────────────────────
    st.markdown('<div class="sb-section">📹 Camera Source</div>', unsafe_allow_html=True)
    camera_mode = st.radio("Mode", ["Upload Videos","CCTV / IP Camera"],
                           key="cam_mode", label_visibility="collapsed")
    cctv_inputs = []
    if camera_mode == "CCTV / IP Camera":
        st.caption("Enter RTSP/HTTP/device URL per lane")
        for i in range(LANES):
            url = st.text_input(f"Lane {i+1} URL",
                                value=CCTV_URLS[i] if i < len(CCTV_URLS) else "",
                                placeholder="rtsp://user:pass@ip/stream",
                                key=f"cctv_{i}")
            cctv_inputs.append(url)

    # ── AI Settings ────────────────────────────────────────────
    st.markdown('<div class="sb-section">🧠 AI Settings</div>', unsafe_allow_html=True)
    conf_thresh  = st.slider("Detection Confidence", 0.10, 0.85, 0.25, 0.05,
                             help="Lower = detect more vehicles, Higher = fewer false positives")
    speed_limit  = st.slider("Speed Limit (km/h)", 20, 120, SPEED_LIMIT, 5)
    ocr_every    = st.slider("OCR Every N Frames",  30, 120, 60, 10)

    # ── Detection Modules ──────────────────────────────────────
    st.markdown('<div class="sb-section">🚨 Detection Modules</div>', unsafe_allow_html=True)
    emergency    = st.toggle("🚑 Emergency / Ambulance Mode", key="emg_tog")
    helmet_check = st.toggle("🪖 Helmet Violation Detection",   value=True)
    if helmet_check:
        helmet_conf = st.slider("  Helmet Conf", 0.10, 0.80, 0.30, 0.05, key="hc")
    else:
        helmet_conf = 0.30
    mobile_check = st.toggle("📵 Mobile Usage Detection",       value=True)
    if mobile_check:
        mobile_conf = st.slider("  Mobile Conf", 0.10, 0.80, 0.35, 0.05, key="mc")
    else:
        mobile_conf = 0.35
    rl_check     = st.toggle("🚨 Red Light Violation",          value=True)
    if rl_check:
        rl_conf = st.slider("  Red Light Conf", 0.10, 0.80, 0.30, 0.05, key="rlc")
    else:
        rl_conf = 0.30
    speed_check  = st.toggle("🚗 Overspeed Detection",          value=True)
    if speed_check:
        speed_conf = st.slider("  Speed Conf (km/h threshold)", 20, 150, int(SPEED_LIMIT), 5, key="sc")
    else:
        speed_conf = SPEED_LIMIT
    ped_check    = st.toggle("🚶 Pedestrian Safety Detection",   value=True)
    if ped_check:
        ped_conf = st.slider("  Ped Sensitivity", 100, 1000, 400, 50, key="pc",
                             help="Pixel count threshold — lower = more sensitive")
    else:
        ped_conf = 400

    # ── Save sidebar values to session_state for fragment access ──
    st.session_state["_conf_thresh"]  = conf_thresh
    st.session_state["_speed_limit"]  = speed_limit
    st.session_state["_ocr_every"]    = ocr_every
    st.session_state["_emergency"]    = emergency
    st.session_state["_helmet_check"] = helmet_check
    st.session_state["_helmet_conf"]  = helmet_conf
    st.session_state["_mobile_check"] = mobile_check
    st.session_state["_rl_check"]     = rl_check
    st.session_state["_speed_check"]  = speed_check
    st.session_state["_ped_check"]    = ped_check
    st.session_state["ped_conf_val"]  = ped_conf

    # ── Weather ────────────────────────────────────────────────
    st.markdown('<div class="sb-section">🌦 Live Weather</div>', unsafe_allow_html=True)
    wx = get_weather(city)
    st.markdown(f'<div style="background:#111827;border-radius:8px;padding:10px;'
                f'text-align:center;color:#00ffcc;font-size:20px;font-weight:700;">{wx}</div>',
                unsafe_allow_html=True)

    # ── Alert in Sidebar ───────────────────────────────────────
    if st.session_state.alert_active:
        msg = st.session_state.current_alert
        imap = {"Ambulance":"🚑","Siren":"🚑","Red Light":"🚨","Overspeed":"🚗",
                "Helmet":"🪖","Mobile":"📵","Accident":"💥","Pedestrian":"🚶"}
        ico  = next((v for k,v in imap.items() if k in msg), "⚠️")
        cls  = "ambu-card" if ("Ambulance" in msg or "Siren" in msg) else "alert-card"
        st.markdown(f'<div class="{cls}">{ico} {msg}</div>', unsafe_allow_html=True)

    # ── Session Stats ──────────────────────────────────────────
    st.markdown('<div class="sb-section">📊 Session Stats</div>', unsafe_allow_html=True)
    tv = sum(st.session_state.lane_counts)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;">
      <div style="background:#111827;border-radius:6px;padding:8px;text-align:center;">
        <div style="color:#00ffcc;font-weight:700;font-size:18px;">{tv}</div>
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
      <div style="background:#111827;border-radius:6px;padding:8px;text-align:center;">
        <div style="color:#ff8844;font-weight:700;font-size:18px;">{st.session_state.ped_crosswalk_count}</div>
        <div style="color:#334;font-size:10px;">PED EVENTS</div></div>
      <div style="background:#111827;border-radius:6px;padding:8px;text-align:center;">
        <div style="color:#ff4444;font-weight:700;font-size:18px;">{st.session_state.ped_danger_count}</div>
        <div style="color:#334;font-size:10px;">PED DANGERS</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔄 Reset System", use_container_width=True):
        keep = {"sel_state","sel_city","sel_area","sel_junc","cam_mode","emg_tog"}
        for k in [k for k in st.session_state if k not in keep]:
            del st.session_state[k]
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="page-title">🚦 SMART CITY TRAFFIC CONTROL v4.0</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-sub">{junction} &nbsp;|&nbsp; {area}, {city} &nbsp;|&nbsp; {LANES} Lanes</div>',
            unsafe_allow_html=True)

show_alert_banner()

# Siren check every 8 s — only during active video detection
if st.session_state.video_started and time.time() - st.session_state.last_siren_check > 8:
    st.session_state.last_siren_check = time.time()
    if detect_siren():
        st.session_state.siren_active = True
        trigger_alert("🚑 Ambulance Siren Detected", lane=st.session_state.ambulance_lane)
    else:
        st.session_state.siren_active = False

if emergency:
    st.session_state.ambulance_active = True
    st.session_state.ambulance_lane   = 0
else:
    # Clear manual override when toggle is turned OFF
    if st.session_state.get("_emergency_was_on"):
        st.session_state.ambulance_active = False
        st.session_state.ambulance_lane   = 0
st.session_state["_emergency_was_on"] = bool(emergency)

# ── AMBULANCE PRIORITY BANNER — only during active detection ──
if st.session_state.video_started and (st.session_state.ambulance_active or st.session_state.siren_active):
    al = st.session_state.ambulance_lane + 1
    st.markdown(f"""
    <div class="amb-banner">
      <div style="font-size:30px;">🚑 🔵 🚑</div>
      <div style="font-family:'Orbitron',monospace;font-size:18px;color:#88aaff;font-weight:700;margin-top:6px;">
        AMBULANCE PRIORITY ACTIVE
      </div>
      <div style="font-size:13px;color:#aabbff;margin-top:6px;">
        Lane {al} → FORCED GREEN &nbsp;|&nbsp; All other lanes → RED &nbsp;|&nbsp; Pedestrian signals → HOLD
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  JUNCTION INTELLIGENCE
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
render_junction_intelligence(junction, city)

# ══════════════════════════════════════════════════════════════
#  VIDEO UPLOAD — NO PAGE REFRESH
# ══════════════════════════════════════════════════════════════

if not st.session_state.video_started:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if camera_mode == "Upload Videos":
        st.subheader("📂 Upload Lane Videos")
        st.info("Upload all lane videos. Detection starts automatically — no page refresh on upload.")

        up_cols = st.columns(LANES)
        for i in range(LANES):
            with up_cols[i]:
                stored = st.session_state.video_bytes.get(f"lane_{i}")
                if stored:
                    st.success(f"✅ Lane {i+1}")
                else:
                    v = st.file_uploader(f"Lane {i+1}", type=["mp4","avi","mpeg4","mov"],
                                         key=f"up_{i}", label_visibility="collapsed")
                    if v is not None:
                        st.session_state.video_bytes[f"lane_{i}"] = v.read()

        ready = sum(1 for i in range(LANES) if st.session_state.video_bytes.get(f"lane_{i}"))
        prog_cols = st.columns(LANES)
        all_ready = True
        for i in range(LANES):
            ok = bool(st.session_state.video_bytes.get(f"lane_{i}"))
            if not ok: all_ready = False
            prog_cols[i].markdown(
                f'<div style="background:{"#003322" if ok else "#1a0a00"};border:1px solid {"#00ee88" if ok else "#4a2a00"};'
                f'border-radius:6px;padding:8px;text-align:center;font-size:12px;'
                f'color:{"#00ee88" if ok else "#7a5a3a"};">{"✅" if ok else "⏳"} Lane {i+1}</div>',
                unsafe_allow_html=True)

        if ready > 0:
            st.markdown(f"**{ready}/{LANES} lanes ready**")

        if all_ready:
            paths = []
            with st.spinner("🔄 Initialising pipeline…"):
                for i in range(LANES):
                    p = f"lane_{i}.mp4"
                    with open(p, "wb") as fh:
                        fh.write(st.session_state.video_bytes[f"lane_{i}"])
                    paths.append(p)
            st.session_state.video_paths   = paths
            st.session_state.video_started = True
            st.session_state.controller    = TrafficSignalController(len(paths))
            st.success(f"✅ {LANES} videos loaded — starting detection!")
            st.rerun()

    elif camera_mode == "CCTV / IP Camera":
        st.subheader("📡 CCTV / IP Camera")
        ca, cb = st.columns(2)
        with ca:
            if st.button("🔗 Connect Cameras", use_container_width=True):
                paths = []
                for i, url in enumerate(cctv_inputs[:LANES]):
                    u = url.strip()
                    if u: paths.append(u)
                    elif Path(f"lane_{i}.mp4").exists(): paths.append(f"lane_{i}.mp4")
                if paths:
                    st.session_state.video_paths   = paths
                    st.session_state.video_started = True
                    st.session_state.controller    = TrafficSignalController(len(paths))
                    st.rerun()
        with cb:
            st.markdown("""<div style="background:#111827;border-radius:8px;padding:12px;font-size:12px;color:#4a6a8a;">
            <b style="color:#00ffcc">Supported:</b><br>rtsp://user:pass@ip/stream<br>http://ip:port/video<br>0/1/2 (webcam)</div>""",
                        unsafe_allow_html=True)

# ── Traffic history ────────────────────────────────────────────
st.session_state.traffic_history.append(sum(st.session_state.lane_counts))
st.session_state.traffic_history = st.session_state.traffic_history[-60:]

# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP  (wrapped in fragment to prevent full-page rerun)
# ══════════════════════════════════════════════════════════════

@st.fragment(run_every=0.35)
def _detection_fragment():
    """Runs every 0.35s, re-renders ONLY the camera feed area."""
    # Pull everything needed from session_state / outer scope
    video_paths   = st.session_state.video_paths
    if not st.session_state.video_started or not video_paths:
        return

    # These values are set in the sidebar — read from session_state inside fragment
    conf_thresh  = st.session_state.get("_conf_thresh", 0.25)
    speed_limit  = st.session_state.get("_speed_limit", 60)
    ocr_every    = st.session_state.get("_ocr_every", 60)
    helmet_check = st.session_state.get("_helmet_check", True)
    helmet_conf  = st.session_state.get("_helmet_conf", 0.30)
    mobile_check = st.session_state.get("_mobile_check", True)
    rl_check     = st.session_state.get("_rl_check", True)
    speed_check  = st.session_state.get("_speed_check", True)
    ped_check    = st.session_state.get("_ped_check", True)
    emergency    = st.session_state.get("_emergency", False)
    junction     = st.session_state.get("sel_junc", "")
    LANES        = len(video_paths)
    controller   = st.session_state.controller

    num_lanes = len(video_paths)
    if controller is None or controller.num_lanes != num_lanes:
        st.session_state.controller = TrafficSignalController(num_lanes)
        controller = st.session_state.controller

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("📹 Live Camera Feeds")

    cam_cols    = st.columns(num_lanes)
    frame_slots = [c.empty() for c in cam_cols]
    stat_cols   = st.columns(num_lanes)
    stat_slots  = [c.empty() for c in stat_cols]

    # ── Persistent VideoCapture: open once, reuse every fragment call ──
    # Release and re-open only when video_paths actually change
    if st.session_state.get("_cap_paths") != list(video_paths):
        for old_cap in st.session_state.get("_caps", []):
            try: old_cap.release()
            except Exception: pass
        new_caps = []
        for path in video_paths:
            try:  src = int(path)
            except (ValueError, TypeError): src = path
            new_caps.append(cv2.VideoCapture(src))
        st.session_state["_caps"]      = new_caps
        st.session_state["_cap_paths"] = list(video_paths)
    caps = st.session_state["_caps"]

    # Carry forward last known counts so skip frames don't show 0
    lane_counts  = list(st.session_state.lane_counts) if st.session_state.lane_counts else [0]*num_lanes
    if len(lane_counts) < num_lanes: lane_counts += [0]*(num_lanes-len(lane_counts))
    lane_counts  = lane_counts[:num_lanes]
    bus_count    = st.session_state.bus_count
    vtype_counts = {cls: 0 for cls in ALL_VEHICLE_CLASSES}

    st.session_state.frame_skip = st.session_state.get("frame_skip", 0) + 1
    # Run full YOLO inference every 3rd fragment call (~1fps) to reduce CPU
    do_infer = (st.session_state.frame_skip % 3 == 0)

    if "_trackers" not in st.session_state:
        st.session_state["_trackers"] = [
            DeepSort(max_age=25, n_init=2, max_cosine_distance=0.35)
            for _ in range(num_lanes)]
    trackers = st.session_state["_trackers"]
    while len(trackers) < num_lanes:
        trackers.append(DeepSort(max_age=25, n_init=2, max_cosine_distance=0.35))

    prev_positions = st.session_state.prev_positions

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if frame is None:
            frame_slots[i].warning(f"⚠️ Lane {i+1}: feed lost")
            continue  # keep cap open; it will retry next fragment call

        disp  = cv2.resize(frame, (640, 360))
        infer = cv2.resize(frame, (416, 416))
        sx, sy = 640/416, 360/416

        # On skip frames carry forward last vehicle count for this lane
        vehicle_count  = lane_counts[i] if not do_infer else 0
        lane_bus_count = 0
        detections     = []
        light_now      = controller.get_light(i)

        if do_infer:
            st.session_state.frame_count += 1
            try:
                results = model(infer, conf=conf_thresh, imgsz=416, verbose=False)
            except Exception as e:
                st.warning(f"Lane {i+1} error: {e}")
                continue  # keep cap open

            for box in results[0].boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                cs     = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = model.names[cls_id].lower().strip()
                if cs < conf_thresh:
                    continue

                x1 = max(0,min(x1,415)); y1 = max(0,min(y1,415))
                x2 = max(0,min(x2,416)); y2 = max(0,min(y2,416))
                dx1,dy1,dx2,dy2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                roi = infer[int(y1):int(y2), int(x1):int(x2)]

                # ── AMBULANCE ─────────────────────────────────────
                # Strict two-gate check:
                # Gate 1: confidence must be >= 0.55 (ambulances are large, clear objects)
                # Gate 2: ROI visual check via is_ambulance_roi() (white body + red cross)
                if label == "ambulance":
                    AMB_CONF_MIN = 0.55
                    roi_ok = is_ambulance_roi(roi) if roi.size > 400 else False
                    # Accept detection only if BOTH gates pass
                    if cs >= AMB_CONF_MIN and roi_ok:
                        if not st.session_state.ambulance_active:
                            st.session_state.ambulance_active = True
                            st.session_state.ambulance_lane   = i
                            trigger_alert("🚑 Ambulance Detected — Priority Cleared", lane=i)
                        cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(50,50,255),4)
                        cv2.putText(disp,f"AMBULANCE {cs:.2f}",(dx1,max(dy1-38,12)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(50,50,255),2)
                    elif cs >= AMB_CONF_MIN and not roi_ok:
                        # YOLO says ambulance but visual check disagrees → treat as regular vehicle
                        label = "car"   # reclassify — don't skip, fall through to normal processing
                    else:
                        # Low confidence ambulance → skip entirely, treat as noise
                        continue
                    if label == "ambulance":
                        vehicle_count += 1
                        if "ambulance" in vtype_counts: vtype_counts["ambulance"] += 1
                        detections.append(([float(x1),float(y1),float(x2-x1),float(y2-y1)],cs,label))
                        continue
                    # If reclassified to car, fall through to normal processing below


                # ── BUS ───────────────────────────────────────────
                if label in BUS_CLASSES:
                    bus_count += 1; lane_bus_count += 1
                    cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,165,255),3)
                    cv2.putText(disp,f"BUS [{label}]",(dx1,max(dy1-10,10)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)
                    st.session_state.bus_log.append({
                        "time":datetime.now().strftime("%H:%M:%S"),
                        "lane":i+1,"type":label,"junction":junction})
                    st.session_state.bus_log = st.session_state.bus_log[-500:]

                # policecar not in trained model — skip

                if label in vtype_counts: vtype_counts[label] += 1
                is_vehicle = (label in ALL_VEHICLE_CLASSES) or (label in MODEL_CLASSES)
                if is_vehicle: vehicle_count += 1

                # ── HELMET ────────────────────────────────────────
                if helmet_check and label in MOTO_CLASSES and roi.size > 100:
                    hy2      = int(y1 + (y2-y1)*0.35)
                    head_roi = infer[int(y1):hy2, int(x1):int(x2)]
                    if not check_helmet(head_roi):
                        trigger_alert("🪖 Helmet Violation", lane=i)
                        cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,165,255),3)
                        cv2.putText(disp,"NO HELMET",(dx1,max(dy1-30,10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,165,255),2)

                # ── OCR ───────────────────────────────────────────
                if (is_vehicle and st.session_state.frame_count % ocr_every == 0 and roi.size > 200):
                    try:
                        with HiddenPrints():
                            ocr_res = reader.readtext(roi, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        for entry in ocr_res:
                            txt = entry[1].upper().replace(" ","")
                            if 5 < len(txt) < 13:
                                if txt not in st.session_state.detected_plates:
                                    st.session_state.detected_plates.append(txt)
                                cv2.putText(disp,txt,(dx1,min(dy2+18,355)),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
                    except Exception:
                        pass

                cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,255,180),2)
                cv2.putText(disp,f"{label} {cs:.2f}",(dx1,max(dy1-6,10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,180),1)
                detections.append(([float(x1),float(y1),float(x2-x1),float(y2-y1)],cs,label))

        # ── DEEPSORT + SPEED + RED LIGHT ──────────────────────────
        try:
            tracks = trackers[i].update_tracks(detections, frame=infer)
            STOP_Y = int(360 * STOP_LINE_FRAC)
            for trk in tracks:
                if not trk.is_confirmed(): continue
                tid = trk.track_id
                tl,tt,tr,tb = trk.to_ltrb()
                tl,tt,tr,tb = int(tl*sx),int(tt*sy),int(tr*sx),int(tb*sy)
                cx,cy = (tl+tr)//2,(tt+tb)//2
                if tid in prev_positions:
                    px,py,pt = prev_positions[tid]
                    dist = np.sqrt((cx-px)**2+(cy-py)**2)
                    dt   = time.time()-pt
                    spd  = int((dist/dt)*PIXEL_TO_KMH) if dt > 0 else 0
                else:
                    spd = 0
                prev_positions[tid] = (cx,cy,time.time())
                scol = (0,0,255) if spd > speed_limit else (0,200,255)
                cv2.putText(disp,f"{spd}km/h",(tl,max(tt-10,10)),cv2.FONT_HERSHEY_SIMPLEX,0.45,scol,1)
                cv2.putText(disp,f"#{tid}",(tl,max(tt-24,10)),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,0),1)
                if rl_check and light_now == "RED" and cy > STOP_Y:
                    trigger_alert("🚨 Red Light Violation", lane=i)
                    cv2.putText(disp,"RED LIGHT CROSS!",(tl,max(tt-40,10)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)
                if speed_check and spd > speed_limit:
                    trigger_alert("🚗 Overspeed", lane=i)
                    cv2.rectangle(disp,(tl,tt),(tr,tb),(0,0,255),3)
        except Exception:
            pass

        # ── PEDESTRIAN SAFETY ─────────────────────────────────────
        if ped_check:
            detect_pedestrian_danger(infer, light_now, i, disp)

        # ── OVERLAYS ──────────────────────────────────────────────
        STOP_Y = int(360 * STOP_LINE_FRAC)
        lc = {"GREEN":(0,220,0),"YELLOW":(0,200,200),"RED":(0,0,220)}.get(light_now,(150,150,150))
        cv2.line(disp,(0,STOP_Y),(640,STOP_Y),lc,2)
        cv2.putText(disp,f"STOP [{light_now}]",(4,STOP_Y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,lc,1)
        cv2.putText(disp,f"LANE {i+1}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,204),2)
        cv2.putText(disp,f"Veh:{vehicle_count}",(8,46),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        if lane_bus_count > 0:
            cv2.putText(disp,f"Bus:{lane_bus_count}",(8,68),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,165,255),1)

        lane_counts[i] = vehicle_count
        frame_slots[i].image(disp, channels="BGR", use_container_width=True)
        stat_slots[i].metric(f"Lane {i+1}", f"{vehicle_count} vehicles")
        # cap stays open — persistent across fragment calls


    # ── Save state ─────────────────────────────────────────────
    st.session_state.lane_counts    = lane_counts
    st.session_state.bus_count      = bus_count
    st.session_state.prev_positions = prev_positions
    st.session_state.last_vtype     = vtype_counts
    st.session_state["sess_total_vehicles"] = (
        st.session_state.get("sess_total_vehicles", 0) + sum(lane_counts))

    total_det = sum(vtype_counts.values())
    if total_det > 0:
        st.session_state.last_conf = min(99.9, (total_det/max(sum(lane_counts),1))*50)

    # ── Update signal ──────────────────────────────────────────
    if not emergency:
        st.session_state.ambulance_active = False
        st.session_state.ambulance_lane   = 0
    amb_lane = st.session_state.ambulance_lane if st.session_state.ambulance_active else -1
    controller.update(lane_counts, ambulance_lane=amb_lane, siren=st.session_state.siren_active)
    if st.session_state.ambulance_active and time.time()-st.session_state.last_alert_time > 20:
        st.session_state.ambulance_active = False
        st.session_state.ambulance_lane   = 0

    # ── FPS ────────────────────────────────────────────────────
    now_t = time.time()
    fps = 1/(now_t - st.session_state.prev_time) if (now_t - st.session_state.prev_time) > 0 else 0
    st.session_state.prev_time = now_t
    st.metric("⚡ FPS", f"{fps:.1f}")

_detection_fragment()

# ── AI Confidence + Vehicle Distribution ───────────────────
vtype_now = st.session_state.last_vtype
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
ca2,cb2 = st.columns(2)
with ca2:
    st.subheader("🤖 AI Confidence")
    st.metric("Model Confidence", f"{st.session_state.last_conf:.1f}%")
    _lc_outer = st.session_state.lane_counts
    if _lc_outer and any(v > 0 for v in _lc_outer):
        st.success(f"✅ Least-congested Lane: Lane {_lc_outer.index(min(_lc_outer))+1}")
with cb2:
    st.subheader("🚗 Vehicle Distribution")
    if sum(vtype_now.values()) > 0:
        grouped = {
            "Car":           vtype_now.get("car", 0),
            "Auto/Tempo":    sum(vtype_now.get(c,0) for c in ["auto","tempo"]),
            "Two Wheelers":  vtype_now.get("two_wheelers", 0),
            "Bicycle":       vtype_now.get("bicycle", 0),
            "Bus":           vtype_now.get("bus", 0),
            "Truck/Tanker":  sum(vtype_now.get(c,0) for c in ["vehicle_truck","truck_tanker"]),
            "Heavy/Other":   sum(vtype_now.get(c,0) for c in ["concrete_mixer","tractor"]),
            "Ambulance":     vtype_now.get("ambulance", 0),
        }
        grouped = {k:v for k,v in grouped.items() if v > 0}
        if grouped:
            st.bar_chart(pd.DataFrame.from_dict(grouped,orient="index",columns=["count"]),height=200)
    else:
        st.info("Awaiting detections…")

now_t = time.time()
fps = 1/(now_t-st.session_state.prev_time) if (now_t-st.session_state.prev_time)>0 else 0
st.session_state.prev_time = now_t
st.metric("⚡ FPS", f"{fps:.1f}")

# ══════════════════════════════════════════════════════════════
#  TRAFFIC SIGNAL STATUS — DYNAMIC
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🚦 Traffic Signal Status — Live Dynamic")

if st.session_state.video_started and (st.session_state.ambulance_active or st.session_state.siren_active):
    al = st.session_state.ambulance_lane + 1
    st.markdown(f"""
    <div style="background:#000055;border:2px solid #4466ff;border-radius:8px;
                padding:10px 16px;margin-bottom:10px;text-align:center;
                font-weight:700;color:#88aaff;font-size:14px;">
      🚑 AMBULANCE OVERRIDE: Lane {al} FORCED GREEN — All others RED
    </div>""", unsafe_allow_html=True)

n_sig    = len(st.session_state.video_paths) if st.session_state.video_paths else LANES
sig_cols = st.columns(n_sig)

for i in range(n_sig):
    light = controller.get_light(i)
    cnt   = st.session_state.lane_counts[i] if i < len(st.session_state.lane_counts) else 0
    timer = controller.time_remaining(i)
    amb   = (i == st.session_state.ambulance_lane and st.session_state.ambulance_active
             and st.session_state.video_started)

    r  = "#ff3333" if light=="RED"    else "#1a0000"
    y  = "#ffdd00" if light=="YELLOW" else "#1a1200"
    g  = "#00ee00" if light=="GREEN"  else "#001a00"
    rg = "0 0 22px #ff3333aa" if light=="RED"    else "none"
    yg = "0 0 22px #ffdd00aa" if light=="YELLOW" else "none"
    gg = "0 0 22px #00ee00aa" if light=="GREEN"  else "none"
    sc = "#00ee00" if light=="GREEN" else "#ff3333" if light=="RED" else "#ffdd00"
    border = "#5577ff" if amb else "#1e2d40"
    amb_tag = '<div style="color:#5577ff;font-size:11px;font-weight:700;margin:2px 0;">🚑 AMBULANCE</div>' if amb else ""
    timer_str = f"{timer}s" if timer != "—" else "—"

    sig_cols[i].markdown(f"""
    <div class="sig-box" style="border:2px solid {border};">
      <div class="sig-lane">LANE {i+1}</div>
      <div class="sig-count">{cnt} veh</div>{amb_tag}
      <div style="background:#050a10;border-radius:10px;padding:10px 8px;width:66px;margin:8px auto;border:1px solid #1a2535;">
        <div style="width:26px;height:26px;background:{r};border-radius:50%;margin:5px auto;box-shadow:{rg};transition:all 0.3s;"></div>
        <div style="width:26px;height:26px;background:{y};border-radius:50%;margin:5px auto;box-shadow:{yg};transition:all 0.3s;"></div>
        <div style="width:26px;height:26px;background:{g};border-radius:50%;margin:5px auto;box-shadow:{gg};transition:all 0.3s;"></div>
      </div>
      <div style="font-family:'Orbitron',monospace;font-size:14px;font-weight:700;color:{sc};">{light}</div>
      <div style="color:#2a4a6a;font-size:11px;margin-top:3px;">⏱ {timer_str}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PEDESTRIAN SAFETY MONITOR
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🚶 Pedestrian Safety Monitor")

pc1,pc2,pc3,pc4 = st.columns(4)
total_ped    = st.session_state.ped_crosswalk_count
total_danger = st.session_state.ped_danger_count
safe_pct     = max(0, int((1 - total_danger/max(total_ped,1))*100))
j_info_ped   = JUNCTION_RISK.get(junction,{})

for col, (val,lbl,clr,ico) in zip([pc1,pc2,pc3,pc4],[
    (total_ped,  "Crosswalk Events",   "#00ffcc","🚶"),
    (total_danger,"Danger Incidents",  "#ff4444","⚠️"),
    (f"{safe_pct}%","Safety Score",    "#00ee88" if safe_pct>80 else "#ffcc00" if safe_pct>50 else "#ff4444","🛡️"),
    ("ACTIVE" if j_info_ped.get("school") else "CLEAR","School Zone",
     "#ff8844" if j_info_ped.get("school") else "#00ee88",
     "🏫" if j_info_ped.get("school") else "✅"),
]):
    col.markdown(f"""
    <div class="card"><div style="font-size:26px;">{ico}</div>
    <div class="card-val" style="color:{clr};font-size:22px;">{val}</div>
    <div class="card-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

if st.session_state.ped_events:
    st.caption("🕐 Recent Pedestrian Events:")
    ev_cols = st.columns(min(len(st.session_state.ped_events[-4:]),4))
    for j, ev in enumerate(reversed(st.session_state.ped_events[-4:])):
        ev_cols[j].markdown(
            f'<div style="background:#330000;border:1px solid #ff4444;border-radius:6px;'
            f'padding:6px 10px;font-size:11px;color:#ffaaaa;">'
            f'⚠️ <b>Lane {ev["lane"]}</b> · {ev["type"]} · {ev["time"]}</div>',
            unsafe_allow_html=True)
else:
    st.success("✅ No pedestrian danger incidents recorded this session")

# ══════════════════════════════════════════════════════════════
#  BUS SCHEDULE TRACKER  (Modern Metro Board)
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🚌 Live Bus Arrivals at Junction")

bus_routes_at = BUS_ROUTES.get(city,{}).get(junction,[])
if not bus_routes_at:
    st.info(f"No bus route data for {junction}. Add routes to BUS_ROUTES dict in the config.")
else:
    bc1,bc2 = st.columns([1,2])
    with bc1:
        st.markdown(f"""
        <div style="background:#0a1a2e;border:1px solid #00ffcc33;border-radius:8px;padding:10px 12px;margin-bottom:8px;">
          <div style="font-family:'Orbitron',monospace;font-size:9px;color:#00ffcc;letter-spacing:2px;margin-bottom:4px;">
            📍 ARRIVAL POINT
          </div>
          <div style="font-size:13px;font-weight:700;color:#ffffff;">{junction}</div>
          <div style="font-size:11px;color:#4a6a8a;">{area}, {city}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**🕐 Look-ahead Window**")
        start_t = st.time_input("From", value=datetime.now().replace(second=0,microsecond=0).time(), key="bs")
        end_t   = st.time_input("To",   value=(datetime.now()+timedelta(hours=1)).replace(second=0,microsecond=0).time(), key="be")
        f_type  = st.multiselect("Bus Type",["AC","Ordinary","Mini","MTC Express"],
                                  default=["AC","Ordinary","Mini","MTC Express"])
        f_dir   = st.selectbox("Direction", ["All","⬇️ Inbound","⬆️ Outbound"], key="bdir")
        s_route = st.text_input("Search Route No.", placeholder="e.g. 21C", key="bsr")

    with bc2:
        ss, es = start_t.strftime("%H:%M"), end_t.strftime("%H:%M")
        buses  = get_buses_arriving_at_junction(junction, city, ss, es)
        if f_type: buses = [b for b in buses if b["Type"] in f_type]
        if f_dir != "All": buses = [b for b in buses if b["Direction"] == f_dir]
        if s_route: buses = [b for b in buses if s_route.upper() in b["Route"].upper()]

        # Next arriving bus highlight
        arriving_soon = [b for b in buses if b["Mins Away"] <= 5]
        if arriving_soon:
            st.markdown(f"""
            <div style="background:linear-gradient(90deg,#0a2a00,#003300);border:1px solid #00ee88;
                        border-radius:8px;padding:10px 14px;margin-bottom:8px;display:flex;align-items:center;gap:10px;">
              <div style="font-size:24px;">🚌</div>
              <div>
                <div style="font-family:'Orbitron',monospace;font-size:11px;color:#00ee88;font-weight:700;">
                  {len(arriving_soon)} BUS{'ES' if len(arriving_soon)>1 else ''} ARRIVING NOW / SOON
                </div>
                <div style="font-size:12px;color:#aaffaa;margin-top:2px;">
                  Routes: {', '.join(b['Route'] for b in arriving_soon[:5])}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if buses:
            # ── METRO-STYLE DEPARTURE BOARD ────────────────────
            now_buses  = [b for b in buses if b["Mins Away"] <= 2]
            soon_buses = [b for b in buses if 2 < b["Mins Away"] <= 10]
            later_buses= [b for b in buses if b["Mins Away"] > 10]

            # Header stats
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.markdown(f'''<div style="background:#0a1f0a;border:1px solid #00ee88;
                border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#00ee88;">{len(buses)}</div>
                <div style="font-size:10px;color:#4a8a4a;letter-spacing:1px;">TOTAL BUSES</div>
            </div>''', unsafe_allow_html=True)
            hc2.markdown(f'''<div style="background:#1f0a0a;border:1px solid #ff4444;
                border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#ff4444;">{len(now_buses)}</div>
                <div style="font-size:10px;color:#8a4a4a;letter-spacing:1px;">ARRIVING NOW</div>
            </div>''', unsafe_allow_html=True)
            hc3.markdown(f'''<div style="background:#1f1a0a;border:1px solid #ffcc00;
                border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#ffcc00;">{len(soon_buses)}</div>
                <div style="font-size:10px;color:#8a7a4a;letter-spacing:1px;">WITHIN 10 MIN</div>
            </div>''', unsafe_allow_html=True)
            peak_count = sum(1 for b in buses if "Peak" in b.get("Peak Hour",""))
            hc4.markdown(f'''<div style="background:#0a0a1f;border:1px solid #4488ff;
                border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#4488ff;">{peak_count}</div>
                <div style="font-size:10px;color:#4a4a8a;letter-spacing:1px;">PEAK HOUR BUSES</div>
            </div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Type filter
            bus_types = sorted(set(b["Type"] for b in buses))
            sel_types = st.multiselect("Filter Bus Type", bus_types, default=bus_types, key="bus_type_f")
            show_buses = [b for b in buses if b["Type"] in sel_types]

            # Search
            search_rt = st.text_input("🔍 Search Route No.", placeholder="e.g. 21C", key="bus_search")
            if search_rt:
                show_buses = [b for b in show_buses if search_rt.upper() in b["Route"].upper()]

            # Metro departure board header
            st.markdown(f"""
            <div style="background:#04080f;border:1px solid #00ffcc44;border-radius:10px;
                        padding:10px 16px;margin:10px 0 4px;">
              <div style="display:grid;grid-template-columns:70px 1fr 80px 90px 80px;
                          gap:8px;font-size:10px;color:#2a5a7a;font-weight:700;letter-spacing:1px;">
                <div>ROUTE</div><div>DESTINATION</div><div>TYPE</div>
                <div>DEPARTS</div><div>STATUS</div>
              </div>
            </div>""", unsafe_allow_html=True)

            for b in show_buses[:25]:
                m    = b["Mins Away"]
                is_pk= "Peak" in b.get("Peak Hour","")
                if m <= 2:
                    status_bg  = "#2a0000"; status_clr = "#ff4444"
                    status_lbl = "🔴 BOARDING"
                    row_bg     = "#0f0505"
                elif m <= 10:
                    status_bg  = "#2a2000"; status_clr = "#ffcc00"
                    status_lbl = f"🟡 {m} MIN"
                    row_bg     = "#0a0a03"
                else:
                    status_bg  = "#002a10"; status_clr = "#00ee88"
                    status_lbl = f"🟢 {m} MIN"
                    row_bg     = "#030a06"
                dir_icon = "⬇️" if "Arriving" in b["Direction"] else "↔️"
                pk_badge = ('<span style="background:#330000;color:#ff4444;font-size:9px;'
                            'border-radius:3px;padding:1px 5px;margin-left:4px;">PEAK</span>') if is_pk else ""
                type_colors = {"AC":"#00ffcc","Ordinary":"#88aaff","Mini":"#ffcc00","MTC Express":"#ff8844"}
                type_clr = type_colors.get(b["Type"],"#aaaaaa")
                st.markdown(f"""
                <div style="background:{row_bg};border:1px solid #0d2035;border-radius:8px;
                            padding:10px 16px;margin:3px 0;display:grid;
                            grid-template-columns:70px 1fr 80px 90px 80px;
                            gap:8px;align-items:center;">
                  <div style="background:#0d2040;border-radius:6px;padding:5px 8px;text-align:center;
                              font-family:'Orbitron',monospace;font-size:14px;font-weight:700;
                              color:#00ffcc;">{b["Route"]}</div>
                  <div>
                    <div style="font-size:13px;color:#ccddee;font-weight:600;">
                      {dir_icon} {b["From"]} → {b["To"]}{pk_badge}
                    </div>
                    <div style="font-size:10px;color:#3a5a7a;margin-top:2px;">
                      Every {b["Freq (min)"]} min &nbsp;·&nbsp; {b["Direction"]}
                    </div>
                  </div>
                  <div style="text-align:center;">
                    <span style="background:#0a1a2a;border:1px solid {type_clr}44;
                                 color:{type_clr};font-size:10px;border-radius:4px;
                                 padding:2px 6px;">{b["Type"]}</span>
                  </div>
                  <div style="text-align:center;">
                    <div style="font-size:15px;font-weight:700;color:#ffffff;">🕐 {b["Time"]}</div>
                  </div>
                  <div style="text-align:center;">
                    <div style="background:{status_bg};border:1px solid {status_clr};
                                border-radius:6px;padding:4px 6px;font-size:11px;
                                font-weight:700;color:{status_clr};">{status_lbl}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info(f"No buses at {junction} between {ss}–{es} with current filters.")

    if st.session_state.bus_log:
        # Filter bus log to show only buses at THIS junction
        junc_bus_log = [b for b in st.session_state.bus_log if b.get("junction") == junction]
        if junc_bus_log:
            st.markdown(f"**🎥 Buses physically detected at {junction} (live feed):**")
            st.dataframe(pd.DataFrame(junc_bus_log[-20:]),
                         use_container_width=True, height=160, hide_index=True)
        else:
            st.info(f"No buses detected in live feed at {junction} yet.")
    else:
        st.info("No buses detected in video feed yet.")

# ══════════════════════════════════════════════════════════════
#  CONGESTION METRICS
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
lane_counts = st.session_state.lane_counts
bus_count   = st.session_state.bus_count
total_v     = sum(lane_counts)
t_score     = total_v * VEHICLE_WEIGHT + bus_count * BUS_WEIGHT

st.subheader("🚦 Real-Time Congestion")
if t_score < 10:   st.success("🟢 Low Traffic — All lanes flowing normally")
elif t_score < 25: st.warning("🟡 Moderate Traffic — Monitor signals")
else:              st.error("🔴 HIGH CONGESTION — Extended green on busiest lane")

for col,(val,lbl,clr) in zip(st.columns(5),[
    (total_v,"Total Vehicles","#00ffcc"),
    (bus_count,"Buses","#ffaa00"),
    (LANES,"Active Lanes","#88aaff"),
    (len(st.session_state.violations),"Violations","#ff5555"),
    (f"Rs{st.session_state.total_fines}","Total Fines","#ffdd00"),
]):
    col.markdown(f'<div class="card"><div class="card-val" style="color:{clr};">{val}</div>'
                 f'<div class="card-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
ch1,ch2 = st.columns(2)
with ch1:
    st.subheader("📊 Lane Traffic Density")
    tdf = pd.DataFrame({"Lane":[f"L{i+1}" for i in range(len(lane_counts))],"Vehicles":lane_counts})
    st.bar_chart(tdf.set_index("Lane"), height=260)
with ch2:
    st.subheader("📈 Traffic Trend")
    st.line_chart(pd.DataFrame({"Vehicles":st.session_state.traffic_history}), height=260)

st.subheader("🔥 Live Congestion Heatmap")
hist_data = st.session_state.traffic_history[-10:]
if len(hist_data) < 2:
    hmap_matrix = np.array(lane_counts,dtype=float).reshape(1,-1)
    row_labels  = ["Now"]
else:
    n_rows = min(len(hist_data),10)
    tn     = sum(lane_counts)+1
    w      = [c/tn for c in lane_counts] if tn>0 else [0.25]*len(lane_counts)
    hmap_matrix = np.array([[round(hist_data[-(n_rows-r)]*wi) for wi in w] for r in range(n_rows)],dtype=float)
    row_labels  = [f"T-{n_rows-r-1}" if r<n_rows-1 else "Now" for r in range(n_rows)]

hdf2  = pd.DataFrame(hmap_matrix,columns=[f"Lane {i+1}" for i in range(len(lane_counts))],index=row_labels)
fig2  = px_plot.imshow(hdf2,text_auto=True,color_continuous_scale="RdYlGn_r",aspect="auto",
                        labels={"x":"Lane","y":"Time","color":"Vehicles"},zmin=0,
                        zmax=max(MAX_GREEN,float(hmap_matrix.max())+1))
fig2.update_layout(height=280,margin=dict(l=40,r=20,t=20,b=40),
                   paper_bgcolor="#080c14",plot_bgcolor="#080c14",font_color="#aaaaaa",
                   coloraxis_colorbar=dict(tickfont=dict(color="#aaaaaa"),
                                           title=dict(text="Veh",font=dict(color="#aaaaaa")),thickness=12,len=0.8),
                   xaxis=dict(tickfont=dict(color="#00ffcc",size=11),gridcolor="#1a2535"),
                   yaxis=dict(tickfont=dict(color="#aaaaaa",size=10),gridcolor="#1a2535"))
fig2.update_traces(textfont=dict(color="white",size=11))
st.plotly_chart(fig2, use_container_width=True)

bar_cols = st.columns(len(lane_counts))
for i,(cnt,col) in enumerate(zip(lane_counts,bar_cols)):
    lv  = "🔴 HIGH" if cnt>8 else "🟡 MED" if cnt>4 else "🟢 LOW"
    clr = "#ff4444" if cnt>8 else "#ffcc00" if cnt>4 else "#00ee88"
    pct = min(int(cnt/max(max(lane_counts),1)*100),100)
    col.markdown(f'<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:8px;padding:8px 10px;text-align:center;">'
                 f'<div style="font-family:monospace;font-size:11px;color:#00ffcc;">Lane {i+1}</div>'
                 f'<div style="font-size:18px;font-weight:700;color:{clr};">{cnt}</div>'
                 f'<div style="background:#111827;border-radius:4px;height:6px;margin:4px 0;">'
                 f'<div style="background:{clr};width:{pct}%;height:6px;border-radius:4px;transition:width 0.4s;"></div></div>'
                 f'<div style="font-size:9px;color:#334;">{lv}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CO₂ EMISSION ESTIMATOR
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🌿 Lane-wise CO₂ Emission Estimator")

CO2_IDLE, CO2_MOVE, HEAVY_F, MOTO_F = 2.6, 1.8, 3.0, 0.6
lane_co2 = []
for idx,cnt in enumerate(lane_counts):
    sig   = controller.get_light(idx)
    heavy = max(1,int(cnt*0.15)); moto = max(0,int(cnt*0.20)); car = max(0,cnt-heavy-moto)
    base  = CO2_IDLE if sig=="RED" else CO2_MOVE
    lane_co2.append(round(car*base + heavy*base*HEAVY_F + moto*base*MOTO_F, 1))

co2_cols = st.columns(len(lane_counts))
for idx,(v,col) in enumerate(zip(lane_co2,co2_cols)):
    clr = "#ff4444" if v>15 else "#ffcc00" if v>7 else "#00ee88"
    lbl = "🔴 High" if v>15 else "🟡 Medium" if v>7 else "🟢 Low"
    col.markdown(f'<div class="card" style="min-height:110px;"><div style="font-size:20px;">🏭</div>'
                 f'<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#00ffcc;margin:4px 0;">LANE {idx+1} CO₂</div>'
                 f'<div style="font-size:20px;font-weight:700;color:{clr};">{v} g/min</div>'
                 f'<div style="font-size:10px;color:#4a6a8a;">{lbl}</div></div>', unsafe_allow_html=True)

tc = round(sum(lane_co2),1); hkg = round(tc*60/1000,2)
tc_clr = "#ff4444" if tc>40 else "#ffcc00" if tc>20 else "#00ee88"
st.markdown(f'<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:8px;padding:10px 16px;'
            f'margin-top:8px;display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:#4a6a8a;font-size:12px;">Junction Total</span>'
            f'<span style="font-size:18px;font-weight:700;color:{tc_clr};">{tc} g/min ≈ {hkg} kg/hr</span>'
            f'<span style="font-size:10px;color:#2a6a4a;">🌱 Green signal reduces ~30% idle emissions</span>'
            f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  JUNCTION INCIDENT HEATMAP
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🗺️ Junction Incident Heatmap Reporter")

tl_str = datetime.now().strftime("%H:%M")
for viol in st.session_state.violations[-5:]:
    ls = viol.get("Lane","Lane 1")
    try: li = int(ls.replace("Lane","").strip())-1 if ls!="—" else 0
    except: li = 0
    li  = max(0, min(li, len(lane_counts)-1))
    ev  = viol.get("Event","")
    cat = ("Ambulance" if any(k in ev for k in ["Ambulance","Siren","🚑"])
           else "Pedestrian" if any(k in ev for k in ["Pedestrian","Crosswalk","🚶"])
           else "Congestion" if (len(lane_counts)>li and lane_counts[li]>8)
           else "Violation")
    entry = {"lane":li+1,"category":cat,"time":tl_str}
    if entry not in st.session_state["incident_log"]:
        st.session_state["incident_log"].append(entry)

for idx,cnt in enumerate(lane_counts):
    if cnt > 8:
        ce = {"lane":idx+1,"category":"Congestion","time":tl_str}
        if ce not in st.session_state["incident_log"]:
            st.session_state["incident_log"].append(ce)

st.session_state["incident_log"] = st.session_state["incident_log"][-60:]
inc_log = st.session_state["incident_log"]

cat_list = ["Violation","Congestion","Ambulance","Pedestrian"]
lane_list = [f"Lane {i+1}" for i in range(LANES)]
inc_matrix = {cat:[0]*LANES for cat in cat_list}
for e in inc_log:
    ln,cat = e["lane"]-1, e["category"]
    if 0<=ln<LANES and cat in inc_matrix:
        inc_matrix[cat][ln] += 1

inc_df  = pd.DataFrame(inc_matrix, index=lane_list)
inc_fig = px_plot.imshow(inc_df.T, text_auto=True, color_continuous_scale="YlOrRd", aspect="auto",
                          labels={"x":"Lane","y":"Incident Type","color":"Count"}, zmin=0)
inc_fig.update_layout(height=230, margin=dict(l=80,r=20,t=20,b=40),
                      paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#aaaaaa",
                      coloraxis_colorbar=dict(tickfont=dict(color="#aaaaaa"),
                                              title=dict(text="Count",font=dict(color="#aaaaaa")),thickness=12,len=0.8),
                      xaxis=dict(tickfont=dict(color="#00ffcc",size=11),gridcolor="#1a2535"),
                      yaxis=dict(tickfont=dict(color="#aaaaaa",size=10),gridcolor="#1a2535"))
inc_fig.update_traces(textfont=dict(color="white",size=12))
st.plotly_chart(inc_fig, use_container_width=True)

if inc_log:
    st.caption("🕐 Recent incidents (latest first):")
    rc = st.columns(4)
    for j,entry in enumerate(reversed(inc_log[-8:])):
        ico = {"Violation":"🚨","Congestion":"🟠","Ambulance":"🚑","Pedestrian":"🚶"}.get(entry["category"],"⚠️")
        rc[j%4].markdown(f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:6px;'
                         f'padding:5px 8px;font-size:11px;color:#aaaaaa;">'
                         f'{ico} <b style="color:#00ffcc">Lane {entry["lane"]}</b> · {entry["category"]} · {entry["time"]}</div>',
                         unsafe_allow_html=True)
else:
    st.info("✅ No incidents recorded yet.")

# ══════════════════════════════════════════════════════════════
#  GREEN WAVE CORRIDOR OPTIMIZER
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🌊 Green Wave Corridor Optimizer")

sorted_lanes = sorted(range(len(lane_counts)), key=lambda x: lane_counts[x])
gw_rows = []; cum_offset = 0
for rank,li in enumerate(sorted_lanes):
    cnt     = lane_counts[li]
    gt      = controller.green_times[li] if li<len(controller.green_times) else BASE_GREEN
    cl      = min(int(gt/2),cnt)
    gw_rows.append({"Lane":f"Lane {li+1}","Rank":rank+1,"Vehicles":cnt,
                    "Green Time (s)":int(gt),"Offset (s)":cum_offset,"Clearance Est.":cl,
                    "Fuel Saved (L)":round(cl*0.04,2),"Time Saved (m)":round(cl*12/60,1)})
    cum_offset += int(gt)+3+18

seq_str = " → ".join([f"Lane {sorted_lanes[r]+1}" for r in range(len(sorted_lanes))])
tfs = round(sum(r["Fuel Saved (L)"] for r in gw_rows),2)
tts = round(sum(r["Time Saved (m)"] for r in gw_rows),1)

st.markdown(f'<div style="background:linear-gradient(135deg,#071a2c,#0d2940);border:1px solid #00ffcc55;'
            f'border-radius:10px;padding:12px 18px;margin-bottom:12px;">'
            f'<div style="font-family:\'Orbitron\',monospace;font-size:11px;color:#00ffcc;letter-spacing:1px;margin-bottom:6px;">OPTIMAL WAVE SEQUENCE</div>'
            f'<div style="font-size:18px;font-weight:700;color:#aaffcc;">{seq_str}</div>'
            f'<div style="display:flex;gap:24px;margin-top:8px;">'
            f'<span style="font-size:12px;color:#ffcc00;">⛽ Fuel saved: ~{tfs} L/cycle</span>'
            f'<span style="font-size:12px;color:#88aaff;">⏱ Time saved: ~{tts} min/cycle</span>'
            f'<span style="font-size:12px;color:#aaffaa;">🌿 Lower idle emissions per vehicle</span>'
            f'</div></div>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame(gw_rows), use_container_width=True, height=min(50+len(gw_rows)*38,260), hide_index=True)

gw_bar_cols = st.columns(len(gw_rows))
for col,row in zip(gw_bar_cols,gw_rows):
    bc = "#00ffcc" if row["Rank"]==1 else "#88aaff" if row["Rank"]==2 else "#4a6a8a"
    col.markdown(f'<div class="card" style="min-height:90px;padding:10px 8px;">'
                 f'<div style="font-size:10px;font-family:\'Orbitron\',monospace;color:{bc};">{row["Lane"]}</div>'
                 f'<div style="font-size:15px;font-weight:700;color:{bc};">#{row["Rank"]}</div>'
                 f'<div style="font-size:10px;color:#4a6a8a;">Offset: {row["Offset (s)"]}s</div>'
                 f'<div style="font-size:10px;color:#ffaa00;">Green: {row["Green Time (s)"]}s</div>'
                 f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DETECTED NUMBER PLATES
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🔢 Detected Number Plates")
plates = st.session_state.detected_plates
if plates:
    p_cols = st.columns(min(len(plates),6))
    for j,p in enumerate(plates[-18:]):
        p_cols[j%6].markdown(
            f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:6px;'
            f'padding:6px 10px;text-align:center;font-family:monospace;color:#00ffcc;font-size:13px;">{p}</div>',
            unsafe_allow_html=True)
else:
    st.info("No plates detected yet — OCR runs every N frames on vehicles")

# ══════════════════════════════════════════════════════════════
#  VIOLATION / FINE LOG
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🚓 Auto Fine & Violation Log")
viols = st.session_state.violations
if viols:
    vdf = pd.DataFrame(viols)
    st.dataframe(vdf, use_container_width=True, height=260)
    tc1,tc2,tc3 = st.columns(3)
    tc1.error(f"💰 Total Fines: Rs {st.session_state.total_fines}")
    tc2.warning(f"📋 Violations: {len(viols)}")
    tc3.info(f"🔢 Unique Plates: {len(set(r.get('Plate','') for r in viols if r.get('Plate','')!='—'))}")
    st.download_button("⬇️ Download Violation Report (CSV)",
                       vdf.to_csv(index=False).encode(), "violations.csv", "text/csv",
                       use_container_width=True)
else:
    st.info("✅ No violations recorded yet")

# ══════════════════════════════════════════════════════════════
#  LOG APPEND
# ══════════════════════════════════════════════════════════════

try:
    lc_safe = (st.session_state.lane_counts+[0,0,0,0])[:4]
    pd.DataFrame([{"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"total":sum(lc_safe),
                   "lane1":lc_safe[0],"lane2":lc_safe[1],"lane3":lc_safe[2],"lane4":lc_safe[3]}]
    ).to_csv(LOG_FILE, mode="a", header=not Path(LOG_FILE).exists(), index=False)
except Exception:
    pass

# ══════════════════════════════════════════════════════════════
#  AUTO RERUN  — handled by @st.fragment(run_every=0.35) above
# ══════════════════════════════════════════════════════════════
# No full-page rerun needed — the detection fragment reruns itself.

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#1e2d40;font-size:11px;padding:6px 0;">
  Smart City Traffic Control v4.0 &nbsp;|&nbsp;
  YOLOv8 + DeepSORT + EasyOCR &nbsp;|&nbsp;
  Ambulance Priority · Pedestrian Safety · Bus Schedule Tracker · Junction Intelligence · Green Wave
</div>""", unsafe_allow_html=True)