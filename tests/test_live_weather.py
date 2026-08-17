import math

from aviation_weather_ml.live_weather import awc_metar_to_model_row


def test_awc_units_match_iem_training_units() -> None:
    payload = {
        "icaoId": "KORD",
        "obsTime": 1699048260,
        "temp": 10.0,
        "dewp": 0.0,
        "wdir": 230,
        "wspd": 6,
        "visib": "10+",
        "altim": 1013.25,
        "clouds": [
            {
                "cover": "BKN",
                "base": 4000,
            }
        ],
    }

    row = awc_metar_to_model_row(payload)

    station = row["station"].iloc[0]
    temperature = row["tmpf"].iloc[0]
    dew_point = row["dwpf"].iloc[0]
    altimeter = row["alti"].iloc[0]
    visibility = row["vsby"].iloc[0]

    assert station == "ORD"
    assert temperature == 50.0
    assert dew_point == 32.0
    assert math.isclose(altimeter, 29.9213, rel_tol=1e-4)
    assert visibility == 10.0
