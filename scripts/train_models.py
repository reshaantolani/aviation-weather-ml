from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from aviation_weather_ml.config import ARTIFACT_DIR, MODEL_FEATURES
from aviation_weather_ml.evaluation import calculate_metrics, save_confusion_matrix
from aviation_weather_ml.sklearn_models import (
    build_logistic_model,
    build_preprocessor,
    build_random_forest_model,
)
from aviation_weather_ml.torch_model import predict_torch_model, train_torch_model


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.sort_values("valid").reset_index(drop=True)
    split_index = int(len(sorted_frame) * train_fraction)

    train_frame = sorted_frame.iloc[:split_index]
    test_frame = sorted_frame.iloc[split_index:]

    return train_frame, test_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_csv", type=Path)
    parser.add_argument(
        "--torch-epochs",
        type=int,
        default=20,
    )
    arguments = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(
        arguments.processed_csv,
        parse_dates=["valid"],
    )

    train_frame, test_frame = chronological_split(frame)

    train_features = train_frame[MODEL_FEATURES]
    test_features = test_frame[MODEL_FEATURES]

    train_labels = train_frame["target_category_index"]
    train_labels = train_labels.astype(int).to_numpy()

    test_labels = test_frame["target_category_index"]
    test_labels = test_labels.astype(int).to_numpy()

    metrics: dict[str, object] = {}

    persistence_predictions = test_frame["current_category_index"]
    persistence_predictions = persistence_predictions.astype(int).to_numpy()

    persistence_metrics = calculate_metrics(
        test_labels,
        persistence_predictions,
    )
    metrics["persistence"] = persistence_metrics

    logistic_model = build_logistic_model()
    logistic_model.fit(train_features, train_labels)
    logistic_predictions = logistic_model.predict(test_features)

    logistic_metrics = calculate_metrics(
        test_labels,
        logistic_predictions,
    )
    metrics["logistic_regression"] = logistic_metrics

    logistic_path = ARTIFACT_DIR / "logistic_regression.joblib"
    joblib.dump(logistic_model, logistic_path)

    random_forest_model = build_random_forest_model()
    random_forest_model.fit(train_features, train_labels)
    random_forest_predictions = random_forest_model.predict(test_features)

    random_forest_metrics = calculate_metrics(
        test_labels,
        random_forest_predictions,
    )
    metrics["random_forest"] = random_forest_metrics

    random_forest_path = ARTIFACT_DIR / "random_forest.joblib"
    joblib.dump(random_forest_model, random_forest_path)

    torch_preprocessor = build_preprocessor()
    train_transformed = torch_preprocessor.fit_transform(train_features)
    test_transformed = torch_preprocessor.transform(test_features)

    if hasattr(train_transformed, "toarray"):
        train_transformed = train_transformed.toarray()
        test_transformed = test_transformed.toarray()

    train_array = np.asarray(train_transformed, dtype=np.float32)
    test_array = np.asarray(test_transformed, dtype=np.float32)

    torch_result = train_torch_model(
        train_array,
        train_labels,
        epochs=arguments.torch_epochs,
    )
    torch_predictions = predict_torch_model(
        torch_result.model,
        test_array,
    )

    torch_metrics = calculate_metrics(
        test_labels,
        torch_predictions,
    )
    metrics["pytorch_mlp"] = torch_metrics

    torch_preprocessor_path = ARTIFACT_DIR / "torch_preprocessor.joblib"
    joblib.dump(torch_preprocessor, torch_preprocessor_path)

    torch_model_data = {
        "input_size": train_array.shape[1],
        "state_dict": torch_result.model.state_dict(),
    }
    torch_model_path = ARTIFACT_DIR / "pytorch_mlp.pt"
    torch.save(torch_model_data, torch_model_path)

    prediction_sets = {
        "persistence": persistence_predictions,
        "logistic_regression": logistic_predictions,
        "random_forest": random_forest_predictions,
        "pytorch_mlp": torch_predictions,
    }

    for name, predictions in prediction_sets.items():
        title = name.replace("_", " ").title()
        confusion_path = ARTIFACT_DIR / f"confusion_{name}.png"

        save_confusion_matrix(
            test_labels,
            predictions,
            confusion_path,
            title=title,
        )

    metric_summary = {}
    for name, values in metrics.items():
        metric_summary[name] = {
            "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"],
        }

    metrics_path = ARTIFACT_DIR / "metrics.json"
    metrics_text = json.dumps(metrics, indent=2)
    metrics_path.write_text(metrics_text)

    summary_text = json.dumps(metric_summary, indent=2)
    print(summary_text)


if __name__ == "__main__":
    main()
