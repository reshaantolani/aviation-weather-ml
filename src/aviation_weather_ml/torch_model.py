from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


class FlightCategoryMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class TorchTrainingResult:
    model: FlightCategoryMLP
    history: list[float]


def train_torch_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.001,
) -> TorchTrainingResult:
    torch.manual_seed(42)

    features_tensor = torch.tensor(
        train_features,
        dtype=torch.float32,
    )
    labels_tensor = torch.tensor(
        train_labels,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        features_tensor,
        labels_tensor,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = FlightCategoryMLP(input_size=train_features.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    class_counts = np.bincount(
        train_labels,
        minlength=4,
    ).astype(np.float32)
    class_weights = np.zeros_like(class_counts)
    nonzero = class_counts > 0
    class_weights[nonzero] = len(train_labels) / (
        len(class_counts) * class_counts[nonzero]
    )
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
        )
    )

    history: list[float] = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0

        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_function(
                logits,
                batch_labels,
            )
            loss.backward()
            optimizer.step()

            batch_count = len(batch_labels)
            total_loss += loss.item() * batch_count
            sample_count += batch_count

        history.append(total_loss / sample_count)

    return TorchTrainingResult(
        model=model,
        history=history,
    )


def predict_torch_model(
    model: FlightCategoryMLP,
    features: np.ndarray,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(
            features,
            dtype=torch.float32,
        )
        logits = model(tensor)
        return (
            torch.argmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )
