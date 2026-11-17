import requests
import streamlit as st

API_KEY = "4fb44d0dd51984990be5d1c3e6873d81"
CITY = "Hyderabad"  # change to your city

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data.get("weather"):
        main = data["weather"][0]["main"]
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        st.sidebar.info(f"🌦️ Weather: {main} ({desc}), {temp}°C")
        return main
    else:
        st.sidebar.warning("⚠️ Unable to fetch weather")
        return None
