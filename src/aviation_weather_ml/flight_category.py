from __future__ import annotations

import math

import pandas as pd

from aviation_weather_ml.config import (
    CATEGORY_TO_INDEX,
)

CEILING_COVERS = {
    "BKN",
    "OVC",
    "VV",
}


def _to_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def find_ceiling_ft(row: pd.Series) -> float:
    ceiling_candidates: list[float] = []

    for layer_number in range(1, 5):
        cover_key = f"skyc{layer_number}"
        height_key = f"skyl{layer_number}"

        cover = str(row.get(cover_key, "")).upper()
        height = _to_float(row.get(height_key))

        if cover in CEILING_COVERS and not math.isnan(height):
            ceiling_candidates.append(height)

    if not ceiling_candidates:
        return math.nan

    return min(ceiling_candidates)


def classify_flight_category(
    visibility_sm: object,
    ceiling_ft: object,
) -> str | None:
    visibility = _to_float(visibility_sm)
    ceiling = _to_float(ceiling_ft)

    if math.isnan(visibility) and math.isnan(ceiling):
        return None

    if (not math.isnan(ceiling) and ceiling < 500) or (
        not math.isnan(visibility) and visibility < 1
    ):
        return "LIFR"

    if (not math.isnan(ceiling) and ceiling < 1000) or (
        not math.isnan(visibility) and visibility < 3
    ):
        return "IFR"

    if (not math.isnan(ceiling) and ceiling <= 3000) or (
        not math.isnan(visibility) and visibility <= 5
    ):
        return "MVFR"

    return "VFR"


def add_flight_category_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    output["ceiling_ft"] = output.apply(
        find_ceiling_ft,
        axis=1,
    )

    output["flight_category"] = [
        classify_flight_category(
            visibility_sm=visibility,
            ceiling_ft=ceiling,
        )
        for visibility, ceiling in zip(
            output["vsby"],
            output["ceiling_ft"],
        )
    ]

    output["current_category_index"] = output["flight_category"].map(CATEGORY_TO_INDEX)

    return output
