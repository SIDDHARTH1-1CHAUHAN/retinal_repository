from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class PredictionResult:
    disease: str
    severity: str
    confidence: float
    grade: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "severity": self.severity,
            "confidence": self.confidence,
        }

    def to_aux_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["grade"] = self.grade
        return payload


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
        stage1_labels = tuple(stage1_paths.get("labels", STAGE1_LABELS))
        return cls(stage1_model=loaded_stage1, dr_model=dr_model, hr_model=hr_model, stage1_labels=stage1_labels)

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
        payload = {
            "stage1_disease": details["stage1_disease"],
            "stage1_confidence": details["stage1_confidence"],
            "grade": details["grade"],
            "final_prediction": prediction.to_dict(),
        }
        return payload

    def _predict_internal(
        self,
        image_tensor: np.ndarray,
        stage1_output: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
    ) -> tuple[PredictionResult, dict[str, Any]]:
        image_batch = self._prepare_image(image_tensor)
        disease_label, stage1_confidence = self._resolve_stage1_prediction(image_batch, stage1_output)

        if disease_label == "normal":
            result = PredictionResult(
                disease=DISEASE_DISPLAY_NAMES["normal"],
                severity="Grade 0",
                confidence=stage1_confidence,
                grade=0,
            )
            return result, {"stage1_disease": disease_label, "stage1_confidence": stage1_confidence, "grade": 0}

        if disease_label == "dr":
            if self.dr_model is None:
                raise ValueError("DR severity model is not loaded.")
            probabilities = self.dr_model.predict(image_batch, verbose=0)[0]
            decoded = decode_dr_prediction(probabilities)
        elif disease_label == "hr":
            if self.hr_model is None:
                raise ValueError("HR severity model is not loaded.")
            probabilities = self.hr_model.predict(image_batch, verbose=0)[0]
            decoded = decode_hr_prediction(probabilities)
        else:
            raise ValueError(f"Unsupported Stage 1 disease label: {disease_label}")

        result = PredictionResult(
            disease=DISEASE_DISPLAY_NAMES[disease_label],
            severity=decoded["label"],
            confidence=decoded["confidence"],
            grade=int(decoded["grade"]),
        )
        return result, {
            "stage1_disease": disease_label,
            "stage1_confidence": stage1_confidence,
            "grade": int(decoded["grade"]),
        }

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
    ) -> tuple[str, float]:
        if stage1_output is None:
            if self.stage1_model is None:
                raise ValueError(
                    "Stage 1 model output was not provided. Supply a preloaded Stage 1 model, "
                    "or call from_config(..., stage1_custom_objects=..., stage1_loader=...)."
                )
            probabilities = np.asarray(self.stage1_model.predict(image_batch, verbose=0)[0], dtype=np.float32)
            class_index = int(probabilities.argmax())
            return self.stage1_labels[class_index], float(probabilities[class_index])

        if isinstance(stage1_output, Mapping):
            if "disease" in stage1_output:
                disease = str(stage1_output["disease"]).strip().lower()
                confidence = float(stage1_output.get("confidence", 0.0))
                return disease, confidence
            if "probabilities" in stage1_output:
                probabilities = np.asarray(stage1_output["probabilities"], dtype=np.float32)
            else:
                raise ValueError("Stage 1 mapping output must contain either 'disease' or 'probabilities'.")
        else:
            probabilities = np.asarray(stage1_output, dtype=np.float32)

        if probabilities.ndim != 1 or probabilities.shape[0] != len(self.stage1_labels):
            raise ValueError(f"Expected Stage 1 probabilities with length {len(self.stage1_labels)}.")
        class_index = int(probabilities.argmax())
        return self.stage1_labels[class_index], float(probabilities[class_index])


def format_prediction_for_report(prediction: Mapping[str, Any]) -> str:
    confidence = float(prediction["confidence"]) * 100.0
    return (
        f"Disease: {prediction['disease']}\n"
        f"Severity: {prediction['severity']}\n"
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
        "final_prediction_fields": ["disease", "severity", "confidence"],
    }
