from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_history(
    history: Mapping[str, Sequence[float]],
    figures_dir: str | Path,
    prefix: str,
) -> None:
    figures_root = Path(figures_dir)
    figures_root.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("loss", [])) + 1)

    if history.get("accuracy"):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["accuracy"], label="Train Accuracy", linewidth=2)
        if history.get("val_accuracy"):
            plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{prefix} Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_root / f"{prefix}_accuracy.png", dpi=200)
        plt.close()

    if history.get("loss"):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["loss"], label="Train Loss", linewidth=2)
        if history.get("val_loss"):
            plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{prefix} Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_root / f"{prefix}_loss.png", dpi=200)
        plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str,
) -> None:
    path = _ensure_parent(output_path)
    matrix = np.asarray(matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prediction_examples(
    image_paths: Sequence[str],
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    confidences: Sequence[float],
    output_path: str | Path,
    title: str,
    max_examples: int = 9,
) -> None:
    try:
        import matplotlib.image as mpimg
    except ImportError:
        return

    if not image_paths:
        return

    count = min(max_examples, len(image_paths))
    path = _ensure_parent(output_path)
    cols = 3
    rows = int(np.ceil(count / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for index in range(count):
        plt.subplot(rows, cols, index + 1)
        try:
            image = mpimg.imread(image_paths[index])
            plt.imshow(image)
        except Exception:
            plt.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
        plt.axis("off")
        plt.title(
            f"T: {true_labels[index]}\nP: {predicted_labels[index]}\nC: {confidences[index] * 100:.1f}%",
            fontsize=9,
        )
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
