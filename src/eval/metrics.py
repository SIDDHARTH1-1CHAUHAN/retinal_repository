from __future__ import annotations

from typing import Iterable, Sequence

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


def _coerce_positive_labels(
    positive_labels: Iterable[int | str] | None,
    label_names: Sequence[str],
) -> list[int]:
    if not positive_labels:
        return []
    resolved: list[int] = []
    for value in positive_labels:
        if isinstance(value, str):
            try:
                resolved.append(label_names.index(value))
            except ValueError:
                continue
        else:
            resolved.append(int(value))
    return [index for index in resolved if 0 <= index < len(label_names)]


def _binary_metrics_from_matrix(matrix: np.ndarray, positive_index: int, label_name: str) -> dict[str, object]:
    true_positive = float(matrix[positive_index, positive_index])
    false_positive = float(matrix[:, positive_index].sum() - true_positive)
    false_negative = float(matrix[positive_index, :].sum() - true_positive)
    true_negative = float(matrix.sum() - true_positive - false_positive - false_negative)
    sensitivity = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    f1_score = _safe_divide(2.0 * precision * sensitivity, precision + sensitivity)
    accuracy = _safe_divide(true_positive + true_negative, matrix.sum())
    return {
        "label": label_name,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1_score,
        "accuracy": accuracy,
        "support": int(matrix[positive_index, :].sum()),
        "confusion": {
            "tp": int(true_positive),
            "fp": int(false_positive),
            "fn": int(false_negative),
            "tn": int(true_negative),
        },
    }


def _quadratic_weighted_kappa(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    if total == 0.0:
        return 0.0
    num_classes = matrix.shape[0]
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    denominator = float((num_classes - 1) ** 2) if num_classes > 1 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / denominator

    actual = matrix / total
    hist_true = matrix.sum(axis=1)
    hist_pred = matrix.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / (total * total)
    numerator = float((weights * actual).sum())
    denominator_value = float((weights * expected).sum())
    if denominator_value == 0.0:
        return 0.0
    return 1.0 - (numerator / denominator_value)


def _expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, num_bins: int = 10) -> float:
    if probabilities.size == 0:
        return 0.0
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        bucket_accuracy = float(correctness[mask].mean())
        bucket_confidence = float(confidences[mask].mean())
        ece += abs(bucket_accuracy - bucket_confidence) * float(mask.mean())
    return float(ece)


def _multiclass_brier_score(probabilities: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    if probabilities.size == 0:
        return 0.0
    targets = np.eye(num_classes, dtype=np.float64)[y_true.astype(np.int32)]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int] | None = None,
    probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    label_names: Sequence[str] | None = None,
    positive_labels: Iterable[int | str] | None = None,
    ordered: bool = False,
) -> dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=np.int32)
    probabilities_array = np.asarray(probabilities, dtype=np.float64) if probabilities is not None else None

    if probabilities_array is not None:
        if probabilities_array.ndim != 2:
            raise ValueError("probabilities must be a 2D array of shape (N, C)")
        inferred_pred = probabilities_array.argmax(axis=1).astype(np.int32)
        if y_pred is None:
            y_pred_array = inferred_pred
        else:
            y_pred_array = np.asarray(y_pred, dtype=np.int32)
    else:
        if y_pred is None:
            raise ValueError("Either y_pred or probabilities must be provided.")
        y_pred_array = np.asarray(y_pred, dtype=np.int32)

    if y_true_array.shape[0] != y_pred_array.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if label_names:
        names = list(label_names)
        num_classes = len(names)
    elif probabilities_array is not None:
        num_classes = int(probabilities_array.shape[1])
        names = [str(index) for index in range(num_classes)]
    else:
        num_classes = int(max(y_true_array.max(initial=0), y_pred_array.max(initial=0)) + 1)
        names = [str(index) for index in range(num_classes)]

    matrix = confusion_matrix(y_true_array, y_pred_array, num_classes=num_classes)
    total = float(matrix.sum())
    accuracy = _safe_divide(float(np.trace(matrix)), total)

    precision_values: list[float] = []
    recall_values: list[float] = []
    specificity_values: list[float] = []
    f1_values: list[float] = []
    support_values: list[int] = []
    per_class: list[dict[str, object]] = []

    for index in range(num_classes):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        true_negative = float(matrix.sum() - true_positive - false_positive - false_negative)
        support = int(matrix[index, :].sum())

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        specificity = _safe_divide(true_negative, true_negative + false_positive)
        f1_score = _safe_divide(2.0 * precision * recall, precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        specificity_values.append(specificity)
        f1_values.append(f1_score)
        support_values.append(support)
        per_class.append(
            {
                "label": names[index],
                "precision": precision,
                "recall": recall,
                "sensitivity": recall,
                "specificity": specificity,
                "f1": f1_score,
                "support": support,
            }
        )

    weights = np.asarray(support_values, dtype=np.float64)
    weighted_precision = float(np.average(precision_values, weights=weights)) if weights.sum() else 0.0
    weighted_recall = float(np.average(recall_values, weights=weights)) if weights.sum() else 0.0
    weighted_specificity = float(np.average(specificity_values, weights=weights)) if weights.sum() else 0.0
    weighted_f1 = float(np.average(f1_values, weights=weights)) if weights.sum() else 0.0

    positive_indices = _coerce_positive_labels(positive_labels, names)
    binary_views = [_binary_metrics_from_matrix(matrix, index, names[index]) for index in positive_indices]

    report: dict[str, object] = {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall_values)) if recall_values else 0.0,
        "precision_macro": float(np.mean(precision_values)) if precision_values else 0.0,
        "precision_weighted": weighted_precision,
        "recall_macro": float(np.mean(recall_values)) if recall_values else 0.0,
        "recall_weighted": weighted_recall,
        "sensitivity_macro": float(np.mean(recall_values)) if recall_values else 0.0,
        "sensitivity_weighted": weighted_recall,
        "specificity_macro": float(np.mean(specificity_values)) if specificity_values else 0.0,
        "specificity_weighted": weighted_specificity,
        "f1_macro": float(np.mean(f1_values)) if f1_values else 0.0,
        "f1_weighted": weighted_f1,
        "support": int(matrix.sum()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "binary_views": binary_views,
    }

    if ordered:
        report["quadratic_weighted_kappa"] = _quadratic_weighted_kappa(matrix)

    if probabilities_array is not None:
        if probabilities_array.shape[0] != y_true_array.shape[0] or probabilities_array.shape[1] != num_classes:
            raise ValueError("probabilities shape must match y_true length and number of classes")
        confidences = probabilities_array.max(axis=1)
        correctness = y_pred_array == y_true_array
        report["mean_confidence"] = float(confidences.mean()) if confidences.size else 0.0
        report["mean_correct_confidence"] = float(confidences[correctness].mean()) if np.any(correctness) else 0.0
        report["mean_incorrect_confidence"] = float(confidences[~correctness].mean()) if np.any(~correctness) else 0.0
        report["ece"] = _expected_calibration_error(probabilities_array, y_true_array)
        report["brier_score"] = _multiclass_brier_score(probabilities_array, y_true_array, num_classes=num_classes)

    return report
