from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import altair as alt

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
    page_title="Aviation Weather ML",
    page_icon="✈",
    layout="wide",
)

st.title("Aviation Weather Flight Category Predictor")
st.caption(
    "Educational ML project. Predictions are not an official weather "
    "briefing and must not be used for flight-safety decisions."
)

model_path = ARTIFACT_DIR / "random_forest.joblib"

if not model_path.exists():
    st.error(f"Train the models first. Expected artifact: {model_path}")
    st.stop()

model = joblib.load(model_path)

icao_id = st.text_input(
    "Airport ICAO identifier",
    value="KORD",
)
icao_id = icao_id.strip().upper()

if st.button("Fetch METAR and predict"):
    try:
        payload = fetch_current_metar(icao_id)
        feature_row = awc_metar_to_model_row(payload)
        model_features = feature_row[MODEL_FEATURES]

        prediction_values = model.predict(model_features)
        prediction = int(prediction_values[0])

        probability_values = model.predict_proba(model_features)
        raw_probabilities = probability_values[0]

        classifier = model.named_steps["classifier"]
        model_classes = classifier.classes_

        probabilities = [0.0] * len(FLIGHT_CATEGORIES)
        for class_index, probability in zip(
            model_classes,
            raw_probabilities,
        ):
            probabilities[int(class_index)] = float(probability)

        predicted_category = FLIGHT_CATEGORIES[prediction]

        current_index = feature_row["current_category_index"].iloc[0]
        current_category = FLIGHT_CATEGORIES[int(current_index)]

        first_column, second_column = st.columns(2)
        first_column.metric("Current category", current_category)
        second_column.metric("Predicted +1 hour", predicted_category)

        probability_data = {
            "category": FLIGHT_CATEGORIES,
            "probability": probabilities,
        }

        probability_frame = pd.DataFrame(probability_data)

        probability_frame["percentage"] = (
            probability_frame["probability"] * 100
        )

        category_colors = {
            "VFR": "#00A651",
            "MVFR": "#0070C0",
            "IFR": "#E31B23",
            "LIFR": "#B000B5",
        }

        highest_probability = probability_frame[
            "percentage"
        ].max()

        chart_max = highest_probability + 10

        if chart_max > 100:
            chart_max = 100

        if chart_max < 20:
            chart_max = 20

        bars = (
            alt.Chart(probability_frame)
            .mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5,
            )
            .encode(
                x=alt.X(
                    "category:N",
                    sort=FLIGHT_CATEGORIES,
                    title=None,
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=14,
                    ),
                ),
                y=alt.Y(
                    "percentage:Q",
                    title="Probability",
                    scale=alt.Scale(
                        domain=[0, chart_max]
                    ),
                    axis=alt.Axis(
                        labelExpr="datum.value + '%'",
                        labelFontSize=12,
                    ),
                ),
                color=alt.Color(
                    "category:N",
                    scale=alt.Scale(
                        domain=[
                            "VFR",
                            "MVFR",
                            "IFR",
                            "LIFR",
                        ],
                        range=[
                            category_colors["VFR"],
                            category_colors["MVFR"],
                            category_colors["IFR"],
                            category_colors["LIFR"],
                        ],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip(
                        "category:N",
                        title="Category",
                    ),
                    alt.Tooltip(
                        "percentage:Q",
                        title="Probability",
                        format=".1f",
                    ),
                ],
            )
        )

        labels = (
            alt.Chart(probability_frame)
            .mark_text(
                dy=-10,
                fontSize=15,
                fontWeight="bold",
            )
            .encode(
                x=alt.X(
                    "category:N",
                    sort=FLIGHT_CATEGORIES,
                ),
                y=alt.Y("percentage:Q"),
                text=alt.Text(
                    "percentage:Q",
                    format=".1f",
                ),
            )
        )

        probability_chart = (
            bars + labels
        ).properties(
            height=400
        )

        st.subheader("Model probabilities")

        st.altair_chart(
            probability_chart,
            use_container_width=True,
        )

        st.subheader("Model input")
        st.dataframe(feature_row, use_container_width=True)

        st.subheader("Raw METAR")
        raw_metar = payload.get("rawOb", payload)
        st.code(str(raw_metar))

    except Exception as error:
        st.exception(error)
