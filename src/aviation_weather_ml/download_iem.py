from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests

from aviation_weather_ml.config import RAW_DIR

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/" "cgi-bin/request/asos.py"

IEM_FIELDS = [
    "tmpf",
    "dwpf",
    "drct",
    "sknt",
    "vsby",
    "alti",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "wxcodes",
]


def normalize_iem_station(
    station: str,
) -> str:
    normalized = station.strip().upper()

    if len(normalized) == 4 and normalized.startswith("K"):
        return normalized[1:]

    return normalized


def download_iem_history(
    stations: list[str],
    start_date: date,
    end_date: date,
    output_path: Path | None = None,
) -> Path:
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_stations = [normalize_iem_station(station) for station in stations]

    if output_path is None:
        station_label = "-".join(normalized_stations)
        output_path = RAW_DIR / (f"iem_{station_label}_" f"{start_date}_{end_date}.csv")

    params: list[tuple[str, str]] = []

    for field in IEM_FIELDS:
        params.append(("data", field))

    for station in normalized_stations:
        params.append(("station", station))

    params.extend(
        [
            (
                "sts",
                f"{start_date.isoformat()}T00:00:00Z",
            ),
            (
                "ets",
                f"{end_date.isoformat()}T00:00:00Z",
            ),
            ("tz", "UTC"),
            ("format", "onlycomma"),
            ("missing", "empty"),
            ("trace", "empty"),
            ("report_type", "3"),
        ]
    )

    response = requests.get(
        IEM_ASOS_URL,
        params=params,
        timeout=120,
        headers={"User-Agent": ("aviation-weather-ml/0.1 " "educational-project")},
    )
    response.raise_for_status()

    output_path.write_bytes(response.content)

    frame = pd.read_csv(output_path)
    if frame.empty:
        raise RuntimeError("IEM returned no observations.")

    return output_path
