import pandas as pd
from datetime import datetime
import streamlit as st
from gtfs_data import GTFSHandler


class DataIntegrator:
    def __init__(self):
        self.gtfs = GTFSHandler()
        self.data = pd.DataFrame(columns=[
            "timestamp",
            "lane_counts",
            "weather",
            "ambulance_detected",
            "signal_lane",
            "signal_status",
            "next_bus",
            "driver_emotion"
        ])

    def log_event(self, lane_counts, weather, ambulance, signal_lane, signal_status, driver_emotion=None):
        next_bus = self.gtfs.get_next_bus("Main Street")  # Example GTFS stop

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lane_counts": lane_counts,
            "weather": weather,
            "ambulance_detected": ambulance,
            "signal_lane": signal_lane,
            "signal_status": signal_status,
            "next_bus": next_bus,
            "driver_emotion": driver_emotion
        }

        self.data = pd.concat([self.data, pd.DataFrame([record])], ignore_index=True)

    def display_log(self):
        return self.data.tail(10)
