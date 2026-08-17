from __future__ import annotations

import numpy as np
import pandas as pd

from aviation_weather_ml.config import (
    CATEGORY_TO_INDEX,
    MODEL_FEATURES,
)
from aviation_weather_ml.flight_category import (
    add_flight_category_columns,
)

NUMERIC_SOURCE_COLUMNS = [
    "tmpf",
    "dwpf",
    "drct",
    "sknt",
    "vsby",
    "alti",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
]


def _clean_numeric_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in NUMERIC_SOURCE_COLUMNS:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce",
            )

    return output


def _add_cyclical_time_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    hour = output["valid"].dt.hour
    month = output["valid"].dt.month

    output["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    output["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    output["month_sin"] = np.sin(2 * np.pi * month / 12)
    output["month_cos"] = np.cos(2 * np.pi * month / 12)

    return output


def prepare_hourly_dataset(
    raw_frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "station",
        "valid",
        "vsby",
    }
    missing = required - set(raw_frame.columns)

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    frame = raw_frame.copy()
    frame["station"] = frame["station"].astype(str).str.upper()
    frame["valid"] = pd.to_datetime(
        frame["valid"],
        utc=True,
        errors="coerce",
    )
    frame = frame.dropna(subset=["valid"])
    frame = _clean_numeric_columns(frame)
    frame = frame.sort_values(["station", "valid"])

    hourly = (
        frame.set_index("valid")
        .groupby("station")
        .resample(
            "1h",
            include_groups=False,
        )
        .last()
        .drop(columns=["station"], errors="ignore")
        .reset_index()
    )

    hourly = add_flight_category_columns(hourly)
    hourly = _add_cyclical_time_features(hourly)

    hourly["target_category"] = hourly.groupby("station")["flight_category"].shift(-1)

    hourly["target_category_index"] = hourly["target_category"].map(CATEGORY_TO_INDEX)

    hourly = hourly.dropna(
        subset=[
            "flight_category",
            "current_category_index",
            "target_category",
            "target_category_index",
        ]
    )

    available_features = [
        column for column in MODEL_FEATURES if column in hourly.columns
    ]

    return hourly[
        [
            "valid",
            "flight_category",
            "target_category",
            "target_category_index",
        ]
        + available_features
    ].copy()
