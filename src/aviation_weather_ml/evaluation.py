from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from aviation_weather_ml.config import FLIGHT_CATEGORIES


def calculate_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    accuracy = accuracy_score(truth, predictions)
    macro_f1 = f1_score(
        truth,
        predictions,
        average="macro",
        zero_division=0,
    )

    labels = list(range(len(FLIGHT_CATEGORIES)))
    report = classification_report(
        truth,
        predictions,
        labels=labels,
        target_names=FLIGHT_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }

    return metrics


def save_confusion_matrix(
    truth: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        truth,
        predictions,
        display_labels=FLIGHT_CATEGORIES,
        ax=axis,
        colorbar=False,
    )

    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_metrics(
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    text = json.dumps(metrics, indent=2)
    output_path.write_text(text)
