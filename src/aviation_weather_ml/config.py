from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_STATIONS = [
    "ORD",
    "SFO",
    "LAX",
    "DEN",
    "DFW",
    "JFK",
    "SEA",
    "ATL",
]

FLIGHT_CATEGORIES = [
    "VFR",
    "MVFR",
    "IFR",
    "LIFR",
]

CATEGORY_TO_INDEX = {
    category: index for index, category in enumerate(FLIGHT_CATEGORIES)
}

INDEX_TO_CATEGORY = {index: category for category, index in CATEGORY_TO_INDEX.items()}

NUMERIC_FEATURES = [
    "tmpf",
    "dwpf",
    "drct",
    "sknt",
    "vsby",
    "alti",
    "ceiling_ft",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "current_category_index",
]

CATEGORICAL_FEATURES = [
    "station",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
