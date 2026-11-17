# controller.py
import time
import streamlit as st

class AdaptiveSignalController:
    def __init__(self, min_green=5, max_green=35):
        self.min_green = min_green
        self.max_green = max_green
        self.current_lane = 0
        self.last_switch = time.time()
        self.signal_status = ["RED"] * 4
        self.base_green_time = 15  # default base time

    def adjust_timing(self, factor=1.0):
        """
        Adjusts base green time dynamically based on weather or traffic conditions.
        Example: factor = 1.5 increases green time by 50%.
        """
        self.base_green_time = int(self.base_green_time * factor)
        st.sidebar.info(f"🕒 Signal timing adjusted (x{factor})")

    def update_signal(self, lane_counts, ambulance=False):
        """
        Main adaptive signal control:
        - Distributes green time based on traffic volume.
        - Overrides for ambulance detection.
        """
        if not lane_counts:
            lane_counts = [0, 0, 0, 0]
        total = sum(lane_counts) + 1e-6

        # compute adaptive green durations
        green_durations = [
            max(self.min_green, int(self.max_green * (count / total)))
            for count in lane_counts
        ]

        elapsed = time.time() - self.last_switch

        # ambulance override
        if ambulance:
            st.warning("🚑 Ambulance detected — prioritizing current lane")
            self.signal_status = ["RED"] * 4
            self.signal_status[self.current_lane] = "GREEN"
            st.session_state["signal_info"] = (
                f"Emergency Mode: Lane {self.current_lane+1} → GREEN"
            )
            return self.signal_status

        # switch lane if green time expired
        if elapsed > green_durations[self.current_lane]:
            self.current_lane = (self.current_lane + 1) % 4
            self.last_switch = time.time()

        self.signal_status = ["RED"] * 4
        self.signal_status[self.current_lane] = "GREEN"

        st.session_state["signal_info"] = (
            f"Lane {self.current_lane+1}: GREEN ({green_durations[self.current_lane]} s)"
        )

        return self.signal_status
