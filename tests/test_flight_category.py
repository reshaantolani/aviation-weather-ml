import math

import pandas as pd

from aviation_weather_ml.flight_category import (
    classify_flight_category,
    find_ceiling_ft,
)


def test_vfr_category() -> None:
    assert (
        classify_flight_category(
            10,
            5000,
        )
        == "VFR"
    )


def test_visibility_can_lower_category() -> None:
    assert (
        classify_flight_category(
            2.5,
            5000,
        )
        == "IFR"
    )


def test_lifr_category() -> None:
    assert (
        classify_flight_category(
            10,
            400,
        )
        == "LIFR"
    )


def test_missing_ceiling_can_be_vfr() -> None:
    assert (
        classify_flight_category(
            10,
            math.nan,
        )
        == "VFR"
    )


def test_lowest_broken_or_overcast_is_ceiling() -> None:
    row = pd.Series(
        {
            "skyc1": "SCT",
            "skyl1": 1200,
            "skyc2": "BKN",
            "skyl2": 2200,
            "skyc3": "OVC",
            "skyl3": 3500,
        }
    )

    assert find_ceiling_ft(row) == 2200


def test_missing_visibility_and_ceiling_is_unknown() -> None:
    assert (
        classify_flight_category(
            math.nan,
            math.nan,
        )
        is None
    )
