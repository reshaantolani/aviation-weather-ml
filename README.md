# Aviation Weather Flight Category Prediction

Machine learning project that predicts airport flight conditions one hour ahead as **VFR, MVFR, IFR, or LIFR** using historical METAR weather observations.

I built this project because I am a pilot and wanted to explore whether current weather conditions could be used to predict short-term changes in flight category.

## Models

The project compares:

* Persistence baseline
* Logistic Regression
* Random Forest
* PyTorch neural network

The models use features such as:

* Temperature
* Dew point
* Wind speed and direction
* Visibility
* Ceiling
* Altimeter pressure
* Current flight category
* Hour and month
* Airport identifier

## Results

The models were evaluated using a chronological train/test split on **42,035 held-out observations**.

| Model               |   Accuracy |  Macro F1 |
| ------------------- | ---------: | --------: |
| Persistence         | **91.17%** | **0.762** |
| Logistic Regression |     88.05% |     0.701 |
| Random Forest       |     89.91% |     0.744 |
| PyTorch MLP         |     89.44% |     0.732 |

The persistence baseline performed best overall, showing that flight conditions are highly stable over a one-hour period. The ML models showed different tradeoffs when identifying less common IFR and LIFR conditions.

Confusion matrices and detailed metrics are available in `artifacts/`.

## Streamlit App

The project includes a Streamlit interface that retrieves a current METAR and displays:

* Current flight category
* Predicted category one hour ahead
* Probability for each category
* Current weather values
* Raw METAR

Run it with:

```bash
streamlit run app/streamlit_app.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Training

Download historical weather data:

```bash
python scripts/download_data.py \
  --stations ORD SFO LAX DEN DFW JFK SEA ATL \
  --start 2023-01-01 \
  --end 2026-01-01
```

Prepare the data:

```bash
python scripts/prepare_data.py \
  data/raw/<downloaded-file>.csv
```

Train the models:

```bash
python scripts/train_models.py \
  data/processed/hourly_training_data.csv
```

## Future Work

* Add previous 1–3 hours of weather as features
* Evaluate longer forecast horizons
* Train on more U.S. METAR-reporting airports
* Compare nationwide and airport-specific models

## Disclaimer

This project is for machine learning experimentation and is not intended for operational flight planning.
