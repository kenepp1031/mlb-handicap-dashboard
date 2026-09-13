"""Keyless weather lookups for outdoor/retractable-roof MLB stadiums via Open-Meteo."""
from __future__ import annotations

from dataclasses import dataclass

import requests
from src.feed import get_json

WEATHER_CODES = {
    0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers", 81: "Rain showers",
    82: "Violent showers", 95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ hail",
}
RAIN_CODES = {51, 53, 55, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}


class WeatherFeedError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeatherSnapshot:
    temperature_f: float
    wind_mph: float
    precipitation_prob: float
    condition: str
    is_inclement: bool


def fetch_stadium_weather(lat: float, lon: float) -> WeatherSnapshot:
    data = get_json(
        "https://api.open-meteo.com/v1/forecast",
        WeatherFeedError,
        params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
            "hourly": "precipitation_probability",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "forecast_days": 1,
        },
    )
    current = data.get("current", {})
    code = int(current.get("weather_code", 0))
    hourly_probs = data.get("hourly", {}).get("precipitation_probability", [])
    current_hour = current.get("time", "")[:13]
    hourly_times = data.get("hourly", {}).get("time", [])
    precip_prob = next((p for t, p in zip(hourly_times, hourly_probs)
                        if t[:13] == current_hour and p is not None), 0.0)

    is_inclement = code in RAIN_CODES or precip_prob >= 50 or current.get("wind_speed_10m", 0) >= 20

    return WeatherSnapshot(
        temperature_f=current.get("temperature_2m", 0.0),
        wind_mph=current.get("wind_speed_10m", 0.0),
        precipitation_prob=precip_prob,
        condition=WEATHER_CODES.get(code, "Unknown"),
        is_inclement=is_inclement,
    )
