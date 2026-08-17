import pandas as pd

from aviation_weather_ml.features import (
    prepare_hourly_dataset,
)


def test_target_is_next_hour_category() -> None:
    raw = pd.DataFrame(
        {
            "station": [
                "ORD",
                "ORD",
                "ORD",
            ],
            "valid": [
                "2025-01-01T00:50:00Z",
                "2025-01-01T01:50:00Z",
                "2025-01-01T02:50:00Z",
            ],
            "tmpf": [40, 40, 40],
            "dwpf": [35, 35, 35],
            "drct": [180, 180, 180],
            "sknt": [10, 10, 10],
            "vsby": [10, 4, 0.5],
            "alti": [30, 30, 30],
            "skyc1": [
                "CLR",
                "BKN",
                "OVC",
            ],
            "skyl1": [
                None,
                2000,
                300,
            ],
        }
    )

    prepared = prepare_hourly_dataset(raw)

    assert list(prepared["flight_category"]) == ["VFR", "MVFR"]

    assert list(prepared["target_category"]) == ["MVFR", "LIFR"]


def test_missing_hour_is_not_labeled_vfr() -> None:
    raw = pd.DataFrame(
        {
            "station": ["ORD", "ORD"],
            "valid": [
                "2025-01-01T00:50:00Z",
                "2025-01-01T02:50:00Z",
            ],
            "vsby": [10, 0.5],
            "tmpf": [40, 40],
            "dwpf": [35, 35],
            "drct": [180, 180],
            "sknt": [10, 10],
            "alti": [30, 30],
            "skyc1": ["CLR", "OVC"],
            "skyl1": [None, 300],
        }
    )

    prepared = prepare_hourly_dataset(raw)

    assert prepared.empty
