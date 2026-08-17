from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from aviation_weather_ml.config import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from aviation_weather_ml.evaluation import (
    calculate_metrics,
    save_confusion_matrix,
)
from aviation_weather_ml.sklearn_models import (
    build_logistic_model,
    build_preprocessor,
    build_random_forest_model,
)
from aviation_weather_ml.torch_model import (
    predict_torch_model,
    train_torch_model,
)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.sort_values("valid").reset_index(drop=True)

    split_index = int(len(sorted_frame) * train_fraction)

    return (
        sorted_frame.iloc[:split_index],
        sorted_frame.iloc[split_index:],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "processed_csv",
        type=Path,
    )
    parser.add_argument(
        "--torch-epochs",
        type=int,
        default=20,
    )
    arguments = parser.parse_args()

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_csv(
        arguments.processed_csv,
        parse_dates=["valid"],
    )

    train_frame, test_frame = chronological_split(frame)

    train_features = train_frame[MODEL_FEATURES]
    test_features = test_frame[MODEL_FEATURES]

    train_labels = train_frame["target_category_index"].astype(int).to_numpy()
    test_labels = test_frame["target_category_index"].astype(int).to_numpy()

    metrics: dict[str, object] = {}

    persistence_predictions = (
        test_frame["current_category_index"].astype(int).to_numpy()
    )
    metrics["persistence"] = calculate_metrics(
        test_labels,
        persistence_predictions,
    )

    logistic_model = build_logistic_model()
    logistic_model.fit(
        train_features,
        train_labels,
    )
    logistic_predictions = logistic_model.predict(test_features)
    metrics["logistic_regression"] = calculate_metrics(
        test_labels,
        logistic_predictions,
    )
    joblib.dump(
        logistic_model,
        ARTIFACT_DIR / "logistic_regression.joblib",
    )

    random_forest_model = build_random_forest_model()
    random_forest_model.fit(
        train_features,
        train_labels,
    )
    random_forest_predictions = random_forest_model.predict(test_features)
    metrics["random_forest"] = calculate_metrics(
        test_labels,
        random_forest_predictions,
    )
    joblib.dump(
        random_forest_model,
        ARTIFACT_DIR / "random_forest.joblib",
    )

    torch_preprocessor = build_preprocessor()
    train_transformed = torch_preprocessor.fit_transform(train_features)
    test_transformed = torch_preprocessor.transform(test_features)

    if hasattr(
        train_transformed,
        "toarray",
    ):
        train_transformed = train_transformed.toarray()
        test_transformed = test_transformed.toarray()

    train_array = np.asarray(
        train_transformed,
        dtype=np.float32,
    )
    test_array = np.asarray(
        test_transformed,
        dtype=np.float32,
    )

    torch_result = train_torch_model(
        train_array,
        train_labels,
        epochs=arguments.torch_epochs,
    )
    torch_predictions = predict_torch_model(
        torch_result.model,
        test_array,
    )
    metrics["pytorch_mlp"] = calculate_metrics(
        test_labels,
        torch_predictions,
    )

    joblib.dump(
        torch_preprocessor,
        ARTIFACT_DIR / "torch_preprocessor.joblib",
    )
    torch.save(
        {
            "input_size": (train_array.shape[1]),
            "state_dict": (torch_result.model.state_dict()),
        },
        ARTIFACT_DIR / "pytorch_mlp.pt",
    )

    prediction_sets = {
        "persistence": (persistence_predictions),
        "logistic_regression": (logistic_predictions),
        "random_forest": (random_forest_predictions),
        "pytorch_mlp": torch_predictions,
    }

    for name, predictions in prediction_sets.items():
        save_confusion_matrix(
            test_labels,
            predictions,
            ARTIFACT_DIR / f"confusion_{name}.png",
            title=name.replace(
                "_",
                " ",
            ).title(),
        )

    metric_summary = {
        name: {
            "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"],
        }
        for name, values in metrics.items()
    }

    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        json.dumps(
            metric_summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
