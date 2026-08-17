from __future__ import annotations

import argparse
from datetime import date

from aviation_weather_ml.download_iem import (
    download_iem_history,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stations",
        nargs="+",
        default=[
            "ORD",
            "SFO",
            "LAX",
            "DEN",
            "DFW",
            "JFK",
            "SEA",
            "ATL",
        ],
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    path = download_iem_history(
        stations=arguments.stations,
        start_date=arguments.start,
        end_date=arguments.end,
    )

    print(path)


if __name__ == "__main__":
    main()
