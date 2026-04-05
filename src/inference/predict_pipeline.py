from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import tensorflow as tf

from src.models.stage2_common import load_model, load_yaml_config
from src.models.stage2_dr_vit import DR_CLASS_NAMES, DR_TASK_SPEC, decode_dr_prediction
from src.models.stage2_hr_vit import HR_CLASS_NAMES, HR_TASK_SPEC, decode_hr_prediction

STAGE1_LABELS = ("normal", "dr", "hr")
DISEASE_DISPLAY_NAMES = {
    "normal": "Normal",
    "dr": "Diabetic Retinopathy",
    "hr": "Hypertensive Retinopathy",
}
DEFAULT_STAGE1_LABEL_ORDER_FILENAMES = (
    "label_order.json",
    "label_map.json",
    "labels.json",
)


@dataclass
class PredictionResult:
    disease: str
    severity: str
    grade: int
    confidence: float
    stage1_confidence: float
    stage2_confidence: float | None
    stage1_probabilities: dict[str, float]
    stage2_probabilities: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "severity": self.severity,
            "grade": self.grade,
            "confidence": self.confidence,
            "stage1_confidence": self.stage1_confidence,
            "stage2_confidence": self.stage2_confidence,
            "stage1_probabilities": self.stage1_probabilities,
            "stage2_probabilities": self.stage2_probabilities,
        }


class RetinalDiseasePredictor:
    def __init__(
        self,
        stage1_model: tf.keras.Model | None = None,
        dr_model: tf.keras.Model | None = None,
        hr_model: tf.keras.Model | None = None,
        stage1_labels: tuple[str, ...] = STAGE1_LABELS,
    ) -> None:
        self.stage1_model = stage1_model
        self.dr_model = dr_model
        self.hr_model = hr_model
        self.stage1_labels = tuple(stage1_labels)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/model_stage2.yaml",
        stage1_model: tf.keras.Model | None = None,
        stage1_custom_objects: Mapping[str, Any] | None = None,
        stage1_loader: Callable[[str | Path], tf.keras.Model] | None = None,
    ) -> "RetinalDiseasePredictor":
        config_path = Path(config_path)
        config = load_yaml_config(config_path)
        inference_config = config.get("inference", {})
        stage1_paths = inference_config.get("stage1", {})
        stage2_paths = inference_config.get("stage2", {})

        loaded_stage1 = stage1_model
        if loaded_stage1 is None and stage1_paths.get("model_path"):
            loaded_stage1 = cls.load_stage1_model(
                model_path=stage1_paths["model_path"],
                custom_objects=stage1_custom_objects,
                loader=stage1_loader,
            )

        dr_model = load_model(stage2_paths["dr_model_path"]) if stage2_paths.get("dr_model_path") else None
        hr_model = load_model(stage2_paths["hr_model_path"]) if stage2_paths.get("hr_model_path") else None
        stage1_labels = cls._load_stage1_label_order(config_path=config_path, inference_config=inference_config)
        return cls(stage1_model=loaded_stage1, dr_model=dr_model, hr_model=hr_model, stage1_labels=stage1_labels)

    @staticmethod
    def _load_stage1_label_order(config_path: Path, inference_config: Mapping[str, Any]) -> tuple[str, ...]:
        stage1_config = dict(inference_config.get("stage1", {}))
        candidate_paths: list[Path] = []
        explicit_path = stage1_config.get("label_order_path") or stage1_config.get("label_map_path")
        if explicit_path:
            candidate_paths.append(_resolve_path(config_path, explicit_path))

        model_path_value = stage1_config.get("model_path")
        if model_path_value:
            model_path = _resolve_path(config_path, model_path_value)
            candidate_paths.extend(model_path.parent / filename for filename in DEFAULT_STAGE1_LABEL_ORDER_FILENAMES)

        candidate_paths.extend(
            _resolve_path(config_path, Path("reports/stage1") / filename)
            for filename in DEFAULT_STAGE1_LABEL_ORDER_FILENAMES
        )

        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            labels = payload.get("labels") if isinstance(payload, dict) else None
            if isinstance(labels, list) and labels:
                return tuple(str(label).strip().lower() for label in labels)
        return STAGE1_LABELS

    @staticmethod
    def load_stage1_model(
        model_path: str | Path,
        custom_objects: Mapping[str, Any] | None = None,
        loader: Callable[[str | Path], tf.keras.Model] | None = None,
    ) -> tf.keras.Model | None:
        if loader is not None:
            return loader(model_path)
        if custom_objects is not None:
            return tf.keras.models.load_model(
                model_path,
                custom_objects=dict(custom_objects),
                compile=False,
            )
        return None

    def predict(
        self,
        image_tensor: np.ndarray,
        stage1_output: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        prediction, _ = self._predict_internal(image_tensor=image_tensor, stage1_output=stage1_output)
        return prediction.to_dict()

    def predict_with_details(
        self,
        image_tensor: np.ndarray,
        stage1_output: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        prediction, details = self._predict_internal(image_tensor=image_tensor, stage1_output=stage1_output)
        return {
            **details,
            "final_prediction": prediction.to_dict(),
        }

    def _predict_internal(
        self,
        image_tensor: np.ndarray,
        stage1_output: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
    ) -> tuple[PredictionResult, dict[str, Any]]:
        image_batch = self._prepare_image(image_tensor)
        stage1_disease, stage1_probabilities = self._resolve_stage1_prediction(image_batch, stage1_output)
        stage1_confidence = float(stage1_probabilities.get(stage1_disease, 0.0))

        if stage1_disease == "normal":
            prediction = PredictionResult(
                disease=DISEASE_DISPLAY_NAMES["normal"],
                severity="Grade 0",
                grade=0,
                confidence=stage1_confidence,
                stage1_confidence=stage1_confidence,
                stage2_confidence=None,
                stage1_probabilities=stage1_probabilities,
                stage2_probabilities=None,
            )
            details = {
                "stage1_disease": stage1_disease,
                "stage1_confidence": stage1_confidence,
                "stage2_confidence": None,
                "grade": 0,
            }
            return prediction, details

        if stage1_disease == "dr":
            if self.dr_model is None:
                raise ValueError("DR severity model is not loaded.")
            probabilities = np.asarray(self.dr_model.predict(image_batch, verbose=0)[0], dtype=np.float32)
            decoded = decode_dr_prediction(probabilities)
            stage2_probabilities = {
                DR_CLASS_NAMES[index]: float(probabilities[index]) for index in range(len(DR_CLASS_NAMES))
            }
        elif stage1_disease == "hr":
            if self.hr_model is None:
                raise ValueError("HR severity model is not loaded.")
            probabilities = np.asarray(self.hr_model.predict(image_batch, verbose=0)[0], dtype=np.float32)
            decoded = decode_hr_prediction(probabilities)
            stage2_probabilities = {
                HR_CLASS_NAMES[index]: float(probabilities[index]) for index in range(len(HR_CLASS_NAMES))
            }
        else:
            raise ValueError(f"Unsupported Stage 1 disease label: {stage1_disease}")

        stage2_confidence = float(decoded["confidence"])
        overall_confidence = min(stage1_confidence, stage2_confidence)
        prediction = PredictionResult(
            disease=DISEASE_DISPLAY_NAMES[stage1_disease],
            severity=decoded["label"],
            grade=int(decoded["grade"]),
            confidence=overall_confidence,
            stage1_confidence=stage1_confidence,
            stage2_confidence=stage2_confidence,
            stage1_probabilities=stage1_probabilities,
            stage2_probabilities=stage2_probabilities,
        )
        details = {
            "stage1_disease": stage1_disease,
            "stage1_confidence": stage1_confidence,
            "stage2_confidence": stage2_confidence,
            "grade": int(decoded["grade"]),
        }
        return prediction, details

    def _prepare_image(self, image_tensor: np.ndarray) -> np.ndarray:
        image = np.asarray(image_tensor, dtype=np.float32)
        if image.shape != (224, 224, 3):
            raise ValueError("Expected a preprocessed image tensor with shape (224, 224, 3).")
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("Expected normalized image values in the range [0, 1].")
        return np.expand_dims(image, axis=0)

    def _resolve_stage1_prediction(
        self,
        image_batch: np.ndarray,
        stage1_output: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[str, dict[str, float]]:
        if stage1_output is None:
            if self.stage1_model is None:
                raise ValueError(
                    "Stage 1 model output was not provided. Supply a preloaded Stage 1 model, "
                    "or call from_config(..., stage1_custom_objects=..., stage1_loader=...)."
                )
            probabilities = np.asarray(self.stage1_model.predict(image_batch, verbose=0)[0], dtype=np.float32)
        elif isinstance(stage1_output, Mapping):
            if "probabilities" in stage1_output:
                probabilities = np.asarray(stage1_output["probabilities"], dtype=np.float32)
            elif "stage1_probabilities" in stage1_output:
                raw_probabilities = stage1_output["stage1_probabilities"]
                if isinstance(raw_probabilities, Mapping):
                    probabilities = np.asarray(
                        [float(raw_probabilities.get(label, 0.0)) for label in self.stage1_labels],
                        dtype=np.float32,
                    )
                else:
                    probabilities = np.asarray(raw_probabilities, dtype=np.float32)
            elif "disease" in stage1_output:
                disease = str(stage1_output["disease"]).strip().lower()
                confidence = float(stage1_output.get("confidence", stage1_output.get("stage1_confidence", 0.0)))
                return disease, {label: (confidence if label == disease else 0.0) for label in self.stage1_labels}
            else:
                raise ValueError("Stage 1 mapping output must contain probabilities or a disease label.")
        else:
            probabilities = np.asarray(stage1_output, dtype=np.float32)

        if probabilities.ndim != 1 or probabilities.shape[0] != len(self.stage1_labels):
            raise ValueError(f"Expected Stage 1 probabilities with length {len(self.stage1_labels)}.")
        probabilities = np.clip(probabilities.astype(np.float32), 0.0, None)
        total = float(probabilities.sum())
        if total > 0:
            probabilities = probabilities / total
        class_index = int(probabilities.argmax())
        label = self.stage1_labels[class_index]
        probability_map = {self.stage1_labels[index]: float(probabilities[index]) for index in range(len(self.stage1_labels))}
        return label, probability_map


def _resolve_path(config_path: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return config_path.resolve().parent.parent.joinpath(candidate).resolve()


def format_prediction_for_report(prediction: Mapping[str, Any]) -> str:
    confidence = float(prediction.get("confidence", 0.0)) * 100.0
    grade = prediction.get("grade", "-")
    return (
        f"Disease: {prediction.get('disease', 'unknown')}\n"
        f"Severity: {prediction.get('severity', 'unknown')}\n"
        f"Grade: {grade}\n"
        f"Confidence: {confidence:.0f}%"
    )


def build_stage2_label_metadata() -> dict[str, Any]:
    return {
        "stage1_labels": list(STAGE1_LABELS),
        "stage2_dr_labels": [
            {"grade": grade, "name": name}
            for grade, name in zip(DR_TASK_SPEC.class_values, DR_CLASS_NAMES)
        ],
        "stage2_hr_labels": [
            {"grade": grade, "name": name}
            for grade, name in zip(HR_TASK_SPEC.class_values, HR_CLASS_NAMES)
        ],
        "final_prediction_fields": [
            "disease",
            "severity",
            "grade",
            "confidence",
            "stage1_confidence",
            "stage2_confidence",
            "stage1_probabilities",
            "stage2_probabilities",
        ],
    }
