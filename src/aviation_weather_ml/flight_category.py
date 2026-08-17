from __future__ import annotations

import math

import pandas as pd

from aviation_weather_ml.config import CATEGORY_TO_INDEX

CEILING_COVERS = {"BKN", "OVC", "VV"}


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

        is_ceiling_cover = cover in CEILING_COVERS
        has_height = not math.isnan(height)

        if is_ceiling_cover and has_height:
            ceiling_candidates.append(height)

    if len(ceiling_candidates) == 0:
        return math.nan

    return min(ceiling_candidates)


def classify_flight_category(
    visibility_sm: object,
    ceiling_ft: object,
) -> str | None:
    visibility = _to_float(visibility_sm)
    ceiling = _to_float(ceiling_ft)

    visibility_missing = math.isnan(visibility)
    ceiling_missing = math.isnan(ceiling)

    if visibility_missing and ceiling_missing:
        return None

    low_lifr_ceiling = not ceiling_missing and ceiling < 500
    low_lifr_visibility = not visibility_missing and visibility < 1
    if low_lifr_ceiling or low_lifr_visibility:
        return "LIFR"

    low_ifr_ceiling = not ceiling_missing and ceiling < 1000
    low_ifr_visibility = not visibility_missing and visibility < 3
    if low_ifr_ceiling or low_ifr_visibility:
        return "IFR"

    low_mvfr_ceiling = not ceiling_missing and ceiling <= 3000
    low_mvfr_visibility = not visibility_missing and visibility <= 5
    if low_mvfr_ceiling or low_mvfr_visibility:
        return "MVFR"

    return "VFR"


def add_flight_category_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()

    output["ceiling_ft"] = output.apply(find_ceiling_ft, axis=1)

    flight_categories = []
    for visibility, ceiling in zip(output["vsby"], output["ceiling_ft"]):
        category = classify_flight_category(
            visibility_sm=visibility,
            ceiling_ft=ceiling,
        )
        flight_categories.append(category)

    output["flight_category"] = flight_categories
    output["current_category_index"] = output["flight_category"].map(
        CATEGORY_TO_INDEX
    )

    return output
