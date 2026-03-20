from __future__ import annotations

from typing import Sequence

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_report_from_confusion(
    matrix: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict[str, object]:
    num_classes = int(matrix.shape[0])
    total = float(matrix.sum())
    accuracy = _safe_divide(float(np.trace(matrix)), total)

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    support_values: list[int] = []
    per_class: list[dict[str, object]] = []

    for index in range(num_classes):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1_score = _safe_divide(2.0 * precision * recall, precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)
        support_values.append(support)
        per_class.append(
            {
                "label": label_names[index] if label_names else str(index),
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "support": support,
            }
        )

    weights = np.asarray(support_values, dtype=np.float32)
    weighted_precision = float(np.average(precision_values, weights=weights)) if weights.sum() else 0.0
    weighted_recall = float(np.average(recall_values, weights=weights)) if weights.sum() else 0.0
    weighted_f1 = float(np.average(f1_values, weights=weights)) if weights.sum() else 0.0

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_macro": float(np.mean(recall_values)) if recall_values else 0.0,
        "f1_macro": float(np.mean(f1_values)) if f1_values else 0.0,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
        "support": int(matrix.sum()),
        "per_class": per_class,
    }


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str] | None = None,
) -> dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=np.int32)
    y_pred_array = np.asarray(y_pred, dtype=np.int32)
    if y_true_array.shape[0] != y_pred_array.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if label_names:
        num_classes = len(label_names)
    else:
        num_classes = int(max(y_true_array.max(initial=0), y_pred_array.max(initial=0)) + 1)

    matrix = confusion_matrix(y_true_array, y_pred_array, num_classes=num_classes)
    report = classification_report_from_confusion(matrix, label_names=label_names)
    report["confusion_matrix"] = matrix.tolist()
    return report
