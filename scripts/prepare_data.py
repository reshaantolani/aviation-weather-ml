from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aviation_weather_ml.config import PROCESSED_DIR
from aviation_weather_ml.features import prepare_hourly_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)

    default_output = PROCESSED_DIR / "hourly_training_data.csv"
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
    )

    arguments = parser.parse_args()

    frame = pd.read_csv(arguments.raw_csv)
    prepared = prepare_hourly_dataset(frame)

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    prepared.to_csv(arguments.output, index=False)

    print(f"Saved {len(prepared):,} rows to {arguments.output}")


if __name__ == "__main__":
    main()
