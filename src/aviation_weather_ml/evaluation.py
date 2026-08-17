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

from aviation_weather_ml.config import (
    FLIGHT_CATEGORIES,
)


def calculate_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    return {
        "accuracy": float(
            accuracy_score(
                truth,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": (
            classification_report(
                truth,
                predictions,
                labels=list(range(len(FLIGHT_CATEGORIES))),
                target_names=(FLIGHT_CATEGORIES),
                output_dict=True,
                zero_division=0,
            )
        ),
    }


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
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def save_metrics(
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    output_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )
