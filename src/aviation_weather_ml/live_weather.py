from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from aviation_weather_ml.flight_category import classify_flight_category

AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"
CEILING_COVERS = {"BKN", "OVC", "VV"}


def fetch_current_metar(icao_id: str) -> dict[str, object]:
    normalized = icao_id.strip().upper()

    params = {
        "ids": normalized,
        "format": "json",
    }
    headers = {
        "User-Agent": "aviation-weather-ml/0.1 educational-project"
    }

    response = requests.get(
        AWC_METAR_URL,
        params=params,
        timeout=20,
        headers=headers,
    )

    if response.status_code == 204:
        raise RuntimeError(
            f"No current METAR is available for {normalized}."
        )

    response.raise_for_status()
    payload = response.json()

    if not payload:
        raise RuntimeError(
            "AviationWeather.gov returned an empty METAR response."
        )

    return payload[0]


def _number(payload: dict[str, object], *keys: str) -> float:
    for key in keys:
        value = payload.get(key)

        if value is None:
            continue

        if isinstance(value, str):
            value = value.replace("+", "")

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return math.nan


def _ceiling_from_awc(payload: dict[str, object]) -> float:
    clouds = payload.get("clouds")
    candidates: list[float] = []

    if isinstance(clouds, list):
        for layer in clouds:
            if not isinstance(layer, dict):
                continue

            cover = str(layer.get("cover", "")).upper()
            base = layer.get("base")

            if cover not in CEILING_COVERS:
                continue

            try:
                base_value = float(base)
                candidates.append(base_value)
            except (TypeError, ValueError):
                continue

    vertical_visibility = _number(payload, "vertVis")
    if not math.isnan(vertical_visibility):
        candidates.append(vertical_visibility)

    if len(candidates) == 0:
        return math.nan

    return min(candidates)


def _celsius_to_fahrenheit(value: float) -> float:
    if math.isnan(value):
        return math.nan

    return value * 9 / 5 + 32


def _hpa_to_in_hg(value: float) -> float:
    if math.isnan(value):
        return math.nan

    return value * 0.0295299830714


def awc_metar_to_model_row(payload: dict[str, object]) -> pd.DataFrame:
    station_value = payload.get("icaoId")
    if station_value is None:
        station_value = payload.get("station")
    if station_value is None:
        station_value = "UNKNOWN"

    station = str(station_value).upper()
    if len(station) == 4 and station.startswith("K"):
        station = station[1:]

    visibility = _number(payload, "visib", "visibility")
    ceiling = _ceiling_from_awc(payload)

    current_category = classify_flight_category(
        visibility,
        ceiling,
    )

    category_indexes = {
        "VFR": 0,
        "MVFR": 1,
        "IFR": 2,
        "LIFR": 3,
    }
    category_index = category_indexes[current_category]

    observation_time = payload.get("obsTime")
    if isinstance(observation_time, (int, float)):
        observed_at = datetime.fromtimestamp(
            observation_time,
            tz=timezone.utc,
        )
    else:
        observed_at = datetime.now(timezone.utc)

    hour = observed_at.hour
    month = observed_at.month

    temp_c = _number(payload, "temp")
    dewp_c = _number(payload, "dewp")
    altim_hpa = _number(payload, "altim")

    temp_f = _celsius_to_fahrenheit(temp_c)
    dewp_f = _celsius_to_fahrenheit(dewp_c)
    altim_in_hg = _hpa_to_in_hg(altim_hpa)

    hour_angle = 2 * np.pi * hour / 24
    month_angle = 2 * np.pi * month / 12

    row = {
        "tmpf": temp_f,
        "dwpf": dewp_f,
        "drct": _number(payload, "wdir"),
        "sknt": _number(payload, "wspd"),
        "vsby": visibility,
        "alti": altim_in_hg,
        "ceiling_ft": ceiling,
        "hour_sin": np.sin(hour_angle),
        "hour_cos": np.cos(hour_angle),
        "month_sin": np.sin(month_angle),
        "month_cos": np.cos(month_angle),
        "current_category_index": category_index,
        "station": station,
    }

    return pd.DataFrame([row])
