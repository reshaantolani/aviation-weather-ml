from __future__ import annotations

from datetime import datetime, timezone

import math
import pandas as pd
import requests

from aviation_weather_ml.flight_category import (
    classify_flight_category,
)

AWC_METAR_URL = "https://aviationweather.gov/" "api/data/metar"


def fetch_current_metar(
    icao_id: str,
) -> dict[str, object]:
    normalized = icao_id.strip().upper()

    response = requests.get(
        AWC_METAR_URL,
        params={
            "ids": normalized,
            "format": "json",
        },
        timeout=20,
        headers={"User-Agent": ("aviation-weather-ml/0.1 " "educational-project")},
    )

    if response.status_code == 204:
        raise RuntimeError("No current METAR is available " f"for {normalized}.")

    response.raise_for_status()
    payload = response.json()

    if not payload:
        raise RuntimeError("AviationWeather.gov returned " "an empty METAR response.")

    return payload[0]


def _number(
    payload: dict[str, object],
    *keys: str,
) -> float:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue

        try:
            if isinstance(value, str):
                value = value.replace("+", "")
            return float(value)
        except (TypeError, ValueError):
            continue

    return math.nan


def _ceiling_from_awc(
    payload: dict[str, object],
) -> float:
    clouds = payload.get("clouds")
    candidates: list[float] = []

    if isinstance(clouds, list):
        for layer in clouds:
            if not isinstance(layer, dict):
                continue

            cover = str(layer.get("cover", "")).upper()
            base = layer.get("base")

            if cover not in {
                "BKN",
                "OVC",
                "VV",
            }:
                continue

            try:
                candidates.append(float(base))
            except (TypeError, ValueError):
                pass

    vertical_visibility = _number(
        payload,
        "vertVis",
    )
    if not math.isnan(vertical_visibility):
        candidates.append(vertical_visibility)

    if not candidates:
        return math.nan

    return min(candidates)


def awc_metar_to_model_row(
    payload: dict[str, object],
) -> pd.DataFrame:
    station = str(payload.get("icaoId") or payload.get("station") or "UNKNOWN").upper()

    if len(station) == 4 and station.startswith("K"):
        station = station[1:]

    visibility = _number(
        payload,
        "visib",
        "visibility",
    )
    ceiling = _ceiling_from_awc(payload)

    current_category = classify_flight_category(
        visibility,
        ceiling,
    )

    category_index = {
        "VFR": 0,
        "MVFR": 1,
        "IFR": 2,
        "LIFR": 3,
    }[current_category]

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

    import numpy as np

    temp_c = _number(payload, "temp")
    dewp_c = _number(payload, "dewp")
    altim_hpa = _number(payload, "altim")

    temp_f = temp_c * 9 / 5 + 32 if not math.isnan(temp_c) else math.nan
    dewp_f = dewp_c * 9 / 5 + 32 if not math.isnan(dewp_c) else math.nan
    altim_in_hg = altim_hpa * 0.0295299830714 if not math.isnan(altim_hpa) else math.nan

    row = {
        "tmpf": temp_f,
        "dwpf": dewp_f,
        "drct": _number(payload, "wdir"),
        "sknt": _number(payload, "wspd"),
        "vsby": visibility,
        "alti": altim_in_hg,
        "ceiling_ft": ceiling,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "current_category_index": (category_index),
        "station": station,
    }

    return pd.DataFrame([row])
