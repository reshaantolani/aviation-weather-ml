from __future__ import annotations

import numpy as np
import pandas as pd

from aviation_weather_ml.config import RAW_DIR


def main() -> None:
    random_generator = np.random.default_rng(42)
    stations = ["ORD", "SFO", "LAX", "DEN"]

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
            day_angle = 2 * np.pi * timestamp.hour / 24 + phase
            slow_angle = 2 * np.pi * index / (24 * 8) + phase

            day_cycle = np.sin(day_angle)
            slow_cycle = np.sin(slow_angle)

            visibility = 7.0 + 3.0 * slow_cycle
            visibility += random_generator.normal(0, 1.2)

            if visibility < 0.25:
                visibility = 0.25
            elif visibility > 10.0:
                visibility = 10.0

            ceiling = 3500 + 2600 * slow_cycle
            ceiling += random_generator.normal(0, 850)

            if ceiling < 200:
                ceiling = 200

            if ceiling < 1000:
                cover = "OVC"
            elif ceiling < 3000:
                cover = "BKN"
            else:
                cover = "SCT"

            temperature = 55 + 15 * day_cycle
            temperature += random_generator.normal(0, 4)

            dew_point = 43 + 8 * slow_cycle
            dew_point += random_generator.normal(0, 3)

            wind_direction = 180 + 90 * slow_cycle
            wind_direction += random_generator.normal(0, 25)
            wind_direction = wind_direction % 360

            wind_speed = 9 + 5 * day_cycle
            wind_speed += random_generator.normal(0, 3)
            if wind_speed < 0:
                wind_speed = 0

            altimeter = 29.92 + random_generator.normal(0, 0.12)

            if cover == "BKN" or cover == "OVC":
                first_layer_height = ceiling
            else:
                first_layer_height = None

            row = {
                "station": station,
                "valid": timestamp,
                "tmpf": temperature,
                "dwpf": dew_point,
                "drct": wind_direction,
                "sknt": wind_speed,
                "vsby": visibility,
                "alti": altimeter,
                "skyc1": cover,
                "skyl1": first_layer_height,
                "skyc2": "",
                "skyl2": None,
                "skyc3": "",
                "skyl3": None,
                "skyc4": "",
                "skyl4": None,
                "wxcodes": "",
            }
            rows.append(row)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / "demo_metar.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)

    print(path)


if __name__ == "__main__":
    main()
