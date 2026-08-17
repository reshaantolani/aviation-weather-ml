from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aviation_weather_ml.config import RAW_DIR


def main() -> None:
    rng = np.random.default_rng(42)
    stations = [
        "ORD",
        "SFO",
        "LAX",
        "DEN",
    ]
    times = pd.date_range(
        "2024-01-01",
        periods=24 * 120,
        freq="1h",
        tz="UTC",
    )

    rows: list[dict[str, object]] = []

    for station_index, station in enumerate(stations):
        phase = station_index * 0.6

        for index, timestamp in enumerate(times):
            day_cycle = np.sin(2 * np.pi * timestamp.hour / 24 + phase)
            slow_cycle = np.sin(2 * np.pi * index / (24 * 8) + phase)

            visibility = 7.0 + 3.0 * slow_cycle + rng.normal(0, 1.2)
            visibility = max(
                0.25,
                min(10.0, visibility),
            )

            ceiling = 3500 + 2600 * slow_cycle + rng.normal(0, 850)
            ceiling = max(200, ceiling)

            if ceiling < 1000:
                cover = "OVC"
            elif ceiling < 3000:
                cover = "BKN"
            else:
                cover = "SCT"

            rows.append(
                {
                    "station": station,
                    "valid": timestamp,
                    "tmpf": (55 + 15 * day_cycle + rng.normal(0, 4)),
                    "dwpf": (43 + 8 * slow_cycle + rng.normal(0, 3)),
                    "drct": (180 + 90 * slow_cycle + rng.normal(0, 25)) % 360,
                    "sknt": max(
                        0,
                        9 + 5 * day_cycle + rng.normal(0, 3),
                    ),
                    "vsby": visibility,
                    "alti": (29.92 + rng.normal(0, 0.12)),
                    "skyc1": cover,
                    "skyl1": (
                        ceiling
                        if cover
                        in {
                            "BKN",
                            "OVC",
                        }
                        else None
                    ),
                    "skyc2": "",
                    "skyl2": None,
                    "skyc3": "",
                    "skyl3": None,
                    "skyc4": "",
                    "skyl4": None,
                    "wxcodes": "",
                }
            )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = RAW_DIR / "demo_metar.csv"
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )
    print(path)


if __name__ == "__main__":
    main()
