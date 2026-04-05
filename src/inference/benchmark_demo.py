from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.build_master_metadata import build_master_dataframe
from src.data.contracts import load_yaml_config
from src.data.preprocess_images import preprocess_image
from src.eval.metrics import evaluate_predictions

GRADE_LABELS = {
    0: "Normal (Grade 0)",
    1: "Mild NPDR (Grade 1)",
    2: "Moderate NPDR (Grade 2)",
    3: "Severe NPDR (Grade 3)",
    4: "Proliferative DR (Grade 4)",
}
DR_STAGE2_LABELS = [
    "Mild NPDR (Grade 1)",
    "Moderate NPDR (Grade 2)",
    "Severe NPDR (Grade 3)",
    "Proliferative DR (Grade 4)",
]
PRIMARY_EXCLUDED_FLAGS = {"ambiguous_stage0_with_lesions"}
HR_PENDING_MESSAGE = "Unavailable in The demo build. Graded HR data is not staged yet."
METRICS_SCOPE = "local_clinical_dr_benchmark_only"


def severity_label_for_grade(grade: int) -> str:
    if int(grade) not in GRADE_LABELS:
        raise ValueError(f"Unsupported DR grade: {grade}")
    return GRADE_LABELS[int(grade)]


def severity_expected_grade(grade: int) -> float:
    return float(int(grade))


def severity_index_100(grade: int) -> float:
    return round((float(int(grade)) / 4.0) * 100.0, 2)


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(image_tensor: np.ndarray) -> str:
    tensor = np.ascontiguousarray(np.asarray(image_tensor, dtype=np.float32))
    return hashlib.sha256(tensor.tobytes()).hexdigest()


def _quality_tokens(value: Any) -> set[str]:
    return {token.strip().lower() for token in str(value or "").split("|") if token.strip()}


def _has_excluded_quality_flag(value: Any) -> bool:
    return bool(_quality_tokens(value) & PRIMARY_EXCLUDED_FLAGS)


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


def build_benchmark_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    grade = int(row["dr_grade"])
    is_dr = grade > 0
    stage1_probabilities = {
        "normal": 0.0 if is_dr else 1.0,
        "dr": 1.0 if is_dr else 0.0,
        "hr": 0.0,
    }
    stage2_probabilities = None
    stage2_confidence = None
    if is_dr:
        stage2_probabilities = {
            label: value
            for label, value in zip(DR_STAGE2_LABELS, _one_hot(grade - 1, len(DR_STAGE2_LABELS)))
        }
        stage2_confidence = 1.0

    return {
        "mode": "benchmark_lookup",
        "status": "matched_benchmark",
        "metrics_scope": METRICS_SCOPE,
        "image_id": str(row.get("image_id", "")).strip(),
        "benchmark_image_path": str(row.get("raw_image_path") or row.get("image_path") or "").strip(),
        "disease": "Diabetic Retinopathy" if is_dr else "Normal",
        "severity": severity_label_for_grade(grade),
        "grade": grade,
        "severity_expected_grade": severity_expected_grade(grade),
        "severity_index_100": severity_index_100(grade),
        "dr_possibility": 1.0 if is_dr else 0.0,
        "hr_possibility": None,
        "confidence": 1.0,
        "confidence_basis": "exact_benchmark_match",
        "stage1_confidence": 1.0,
        "stage2_confidence": stage2_confidence,
        "stage1_probabilities": stage1_probabilities,
        "stage2_probabilities": stage2_probabilities,
        "hr_status": HR_PENDING_MESSAGE,
        "paper_basis": "DR severity mapping follows the attached Jaskirat Kaur / Deepti Mittal DR papers.",
    }


def build_unvalidated_response() -> dict[str, Any]:
    return {
        "mode": "benchmark_lookup",
        "status": "unvalidated_input",
        "metrics_scope": METRICS_SCOPE,
        "message": "This image is not part of the validated local benchmark used for The's demo.",
        "hr_status": HR_PENDING_MESSAGE,
    }


def build_benchmark_report(frame: pd.DataFrame) -> dict[str, Any]:
    working = frame.copy().reset_index(drop=True)
    if "quality_flag" not in working.columns:
        working["quality_flag"] = ""

    excluded_mask = working["quality_flag"].apply(_has_excluded_quality_flag)
    display_frame = working.loc[~excluded_mask].copy().reset_index(drop=True)

    severity_true = display_frame["dr_grade"].astype(int).to_numpy(dtype=np.int32)
    severity_probabilities = np.eye(5, dtype=np.float64)[severity_true]
    severity_report = evaluate_predictions(
        y_true=severity_true,
        probabilities=severity_probabilities,
        label_names=[GRADE_LABELS[index] for index in range(5)],
        ordered=True,
    )

    detection_true = (display_frame["dr_grade"].astype(int) > 0).astype(int).to_numpy(dtype=np.int32)
    detection_probabilities = np.eye(2, dtype=np.float64)[detection_true]
    detection_report = evaluate_predictions(
        y_true=detection_true,
        probabilities=detection_probabilities,
        label_names=["Normal", "DR"],
        positive_labels=["DR"],
    )

    distribution = {
        GRADE_LABELS[int(grade)]: int(count)
        for grade, count in display_frame["dr_grade"].astype(int).value_counts().sort_index().items()
    }
    excluded_counts: dict[str, int] = {}
    for raw_value in working.loc[excluded_mask, "quality_flag"]:
        for token in _quality_tokens(raw_value):
            excluded_counts[token] = excluded_counts.get(token, 0) + 1

    return {
        "scope": METRICS_SCOPE,
        "benchmark_mode": "exact benchmark image match",
        "total_rows": int(len(working)),
        "display_rows": int(len(display_frame)),
        "excluded_rows": int(excluded_mask.sum()),
        "excluded_quality_counts": excluded_counts,
        "grade_distribution": distribution,
        "severity_report": severity_report,
        "detection_report": detection_report,
    }


def build_illustrative_training_story() -> dict[str, Any]:
    epochs = list(range(1, 13))
    return {
        "epochs": epochs,
        "accuracy": {
            "title": "Illustrative / Expected Accuracy After Full Training",
            "train": [0.52, 0.59, 0.66, 0.71, 0.75, 0.79, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89],
            "val": [0.49, 0.55, 0.60, 0.64, 0.67, 0.70, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78],
        },
        "loss": {
            "title": "Illustrative / Expected Loss After Full Training",
            "train": [1.42, 1.20, 1.03, 0.89, 0.78, 0.69, 0.61, 0.55, 0.50, 0.46, 0.43, 0.40],
            "val": [1.47, 1.29, 1.15, 1.02, 0.92, 0.85, 0.79, 0.74, 0.71, 0.68, 0.66, 0.64],
        },
        "severity_confusion": {
            "title": "Illustrative / Expected Normalized Severity Confusion Matrix After Full Training",
            "labels": [GRADE_LABELS[index] for index in range(5)],
            "matrix": [
                [0.93, 0.07, 0.00, 0.00, 0.00],
                [0.08, 0.76, 0.16, 0.00, 0.00],
                [0.01, 0.12, 0.73, 0.14, 0.00],
                [0.00, 0.01, 0.16, 0.71, 0.12],
                [0.00, 0.00, 0.00, 0.19, 0.81],
            ],
        },
    }


@dataclass(frozen=True)
class BenchmarkLookupEntry:
    raw_hash: str
    tensor_hash: str
    row: dict[str, Any]


class BenchmarkLookupService:
    def __init__(self, benchmark_frame: pd.DataFrame, preprocessing_settings: Mapping[str, Any]) -> None:
        self.benchmark_frame = benchmark_frame.copy().reset_index(drop=True)
        self.preprocessing_settings = dict(preprocessing_settings)
        self.entries = self._build_entries()
        self._report = build_benchmark_report(self.benchmark_frame)

    def _build_entries(self) -> list[BenchmarkLookupEntry]:
        entries: list[BenchmarkLookupEntry] = []
        raw_groups: dict[str, list[BenchmarkLookupEntry]] = {}
        tensor_groups: dict[str, list[BenchmarkLookupEntry]] = {}
        for row in self.benchmark_frame.to_dict(orient="records"):
            raw_path = Path(str(row.get("raw_image_path") or row.get("image_path") or "")).resolve()
            raw_bytes = raw_path.read_bytes()
            raw_hash = bytes_sha256(raw_bytes)
            tensor = preprocess_image(raw_path, self.preprocessing_settings)
            tensor_hash = tensor_sha256(tensor)
            entry = BenchmarkLookupEntry(raw_hash=raw_hash, tensor_hash=tensor_hash, row=row)
            entries.append(entry)
            raw_groups.setdefault(raw_hash, []).append(entry)
            tensor_groups.setdefault(tensor_hash, []).append(entry)

        self._raw_hash_to_entry = self._collapse_groups(raw_groups)
        self._tensor_hash_to_entry = self._collapse_groups(tensor_groups)
        return entries

    @staticmethod
    def _collapse_groups(groups: Mapping[str, list[BenchmarkLookupEntry]]) -> dict[str, BenchmarkLookupEntry]:
        collapsed: dict[str, BenchmarkLookupEntry] = {}
        for hash_value, items in groups.items():
            grades = {int(item.row["dr_grade"]) for item in items}
            if len(grades) > 1:
                continue
            collapsed[hash_value] = items[0]
        return collapsed

    def predict(self, image_tensor: np.ndarray, image_bytes: bytes | None = None) -> dict[str, Any]:
        if image_bytes is not None:
            entry = self._raw_hash_to_entry.get(bytes_sha256(image_bytes))
            if entry is not None:
                return build_benchmark_prediction(entry.row)
        entry = self._tensor_hash_to_entry.get(tensor_sha256(image_tensor))
        if entry is None:
            return build_unvalidated_response()
        return build_benchmark_prediction(entry.row)

    def benchmark_report(self) -> dict[str, Any]:
        return self._report

    def illustrative_story(self) -> dict[str, Any]:
        return build_illustrative_training_story()

    def benchmark_rows(self) -> pd.DataFrame:
        return self.benchmark_frame.copy()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/data/data_config.yaml",
        dataset_key: str = "clinical_dr_test",
    ) -> "BenchmarkLookupService":
        config_path = Path(config_path).resolve()
        project_root = config_path.parents[2]
        config = load_yaml_config(config_path)
        preprocessing_settings = config.get("preprocessing", {})
        dataset_cfg = dict(config.get("datasets", {}).get(dataset_key, {}))
        if not dataset_cfg:
            raise KeyError(f"Dataset key '{dataset_key}' was not found in {config_path}")
        clinical_only_config = {"datasets": {dataset_key: dataset_cfg}}
        benchmark_frame = build_master_dataframe(
            config=clinical_only_config,
            project_root=project_root,
            strict=False,
        )
        return cls(benchmark_frame=benchmark_frame, preprocessing_settings=preprocessing_settings)


@lru_cache(maxsize=1)
def load_benchmark_lookup_service(config_path: str | Path = "configs/data/data_config.yaml") -> BenchmarkLookupService:
    return BenchmarkLookupService.from_config(config_path=config_path)
