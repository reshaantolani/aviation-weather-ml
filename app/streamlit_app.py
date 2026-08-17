from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviation_weather_ml.config import (  # noqa: E402
    ARTIFACT_DIR,
    FLIGHT_CATEGORIES,
    MODEL_FEATURES,
)
from aviation_weather_ml.live_weather import (  # noqa: E402
    awc_metar_to_model_row,
    fetch_current_metar,
)

st.set_page_config(
    page_title=("Aviation Weather ML"),
    page_icon="✈",
    layout="wide",
)

st.title("Aviation Weather Flight Category Predictor")
st.caption(
    "Educational ML project. Predictions are not "
    "an official weather briefing and must not be "
    "used for flight-safety decisions."
)

model_path = ARTIFACT_DIR / "random_forest.joblib"

if not model_path.exists():
    st.error("Train the models first. Expected artifact: " f"{model_path}")
    st.stop()

model = joblib.load(model_path)

icao_id = (
    st.text_input(
        "Airport ICAO identifier",
        value="KORD",
    )
    .strip()
    .upper()
)

if st.button("Fetch METAR and predict"):
    try:
        payload = fetch_current_metar(icao_id)
        feature_row = awc_metar_to_model_row(payload)
        model_features = feature_row[MODEL_FEATURES]

        prediction = int(model.predict(model_features)[0])
        raw_probabilities = model.predict_proba(model_features)[0]
        model_classes = model.named_steps["classifier"].classes_
        probabilities = [0.0] * len(FLIGHT_CATEGORIES)
        for class_index, probability in zip(
            model_classes,
            raw_probabilities,
        ):
            probabilities[int(class_index)] = float(probability)

        predicted_category = FLIGHT_CATEGORIES[prediction]

        first_column, second_column = st.columns(2)
        first_column.metric(
            "Current category",
            FLIGHT_CATEGORIES[int(feature_row["current_category_index"].iloc[0])],
        )
        second_column.metric(
            "Predicted +1 hour",
            predicted_category,
        )

        probability_frame = pd.DataFrame(
            {
                "category": (FLIGHT_CATEGORIES),
                "probability": (probabilities),
            }
        ).set_index("category")

        st.subheader("Model probabilities")
        st.bar_chart(probability_frame)

        st.subheader("Model input")
        st.dataframe(
            feature_row,
            use_container_width=True,
        )

        st.subheader("Raw METAR")
        st.code(
            str(
                payload.get(
                    "rawOb",
                    payload,
                )
            )
        )

    except Exception as error:
        st.exception(error)
