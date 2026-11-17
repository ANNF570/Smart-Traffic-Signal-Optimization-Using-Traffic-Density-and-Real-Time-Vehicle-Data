import pandas as pd
from datetime import datetime
import os


class GTFSHandler:
    def __init__(self, gtfs_folder="gtfs_data"):
        self.gtfs_folder = gtfs_folder

        try:
            self.stops = pd.read_csv(os.path.join(gtfs_folder, "stops.txt"))
            self.trips = pd.read_csv(os.path.join(gtfs_folder, "trips.txt"))
            self.stop_times = pd.read_csv(os.path.join(gtfs_folder, "stop_times.txt"))
            self.routes = pd.read_csv(os.path.join(gtfs_folder, "routes.txt"))
            self.calendar = pd.read_csv(os.path.join(gtfs_folder, "calendar.txt"))

            print("✅ GTFS data loaded successfully")

        except Exception as e:
            print(f"❌ GTFS load error: {e}")

    # -----------------------------------------------------
    # Check if service is active today
    # -----------------------------------------------------
    def is_service_active_today(self, service_id):
        today = datetime.now()
        weekday = today.strftime("%A").lower()

        service = self.calendar[self.calendar["service_id"] == service_id]

        if service.empty:
            return False

        row = service.iloc[0]

        # Check weekday availability
        if row[weekday] != 1:
            return False

        # Check start and end date
        start_date = datetime.strptime(str(row["start_date"]), "%Y%m%d")
        end_date = datetime.strptime(str(row["end_date"]), "%Y%m%d")

        if not (start_date <= today <= end_date):
            return False

        return True

    # -----------------------------------------------------
    # Get next bus (single best match)
    # -----------------------------------------------------
    def get_next_bus(self, stop_name):

        upcoming = self.get_upcoming_buses(stop_name, limit=1)

        if not upcoming:
            return "No upcoming buses"

        return upcoming[0]

    # -----------------------------------------------------
    # Get multiple upcoming buses
    # -----------------------------------------------------
    def get_upcoming_buses(self, stop_name, limit=5):

        try:
            now = datetime.now()
            current_time = now.time()

            # Find stop
            stop_info = self.stops[
                self.stops["stop_name"].str.contains(stop_name, case=False, na=False)
            ]

            if stop_info.empty:
                return []

            stop_id = stop_info.iloc[0]["stop_id"]

            # Filter stop_times
            stop_times_filtered = self.stop_times[
                self.stop_times["stop_id"] == stop_id
            ].copy()

            results = []

            for _, row in stop_times_filtered.iterrows():

                try:
                    departure_time = datetime.strptime(
                        row["departure_time"], "%H:%M:%S"
                    ).time()
                except:
                    continue

                if departure_time <= current_time:
                    continue

                trip_id = row["trip_id"]
                trip_info = self.trips[self.trips["trip_id"] == trip_id]

                if trip_info.empty:
                    continue

                service_id = trip_info.iloc[0]["service_id"]

                # Validate service active
                if not self.is_service_active_today(service_id):
                    continue

                route_id = trip_info.iloc[0]["route_id"]
                headsign = trip_info.iloc[0]["trip_headsign"]

                route_info = self.routes[self.routes["route_id"] == route_id]

                if not route_info.empty:
                    route_number = route_info.iloc[0]["route_short_name"]
                else:
                    route_number = route_id

                delta_minutes = (
                    datetime.combine(datetime.today(), departure_time)
                    - datetime.combine(datetime.today(), current_time)
                ).seconds // 60

                results.append(
                    f"{route_number} → {headsign} | {delta_minutes} min"
                )

            # Sort by minutes
            results = sorted(
                results,
                key=lambda x: int(x.split("|")[1].replace("min", "").strip())
            )

            return results[:limit]

        except Exception as e:
            print("GTFS Error:", e)
            return []
