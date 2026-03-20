from __future__ import annotations

from typing import Any, Mapping

from src.models.stage2_common import Stage2TaskSpec, build_vit_classifier

HR_CLASS_VALUES = (1, 2, 3, 4)
HR_CLASS_NAMES = (
    "Mild HR (Grade 1)",
    "Moderate HR (Grade 2)",
    "Severe HR (Grade 3)",
    "Malignant HR (Grade 4)",
)

HR_TASK_SPEC = Stage2TaskSpec(
    name="stage2_hr",
    disease_label="hr",
    grade_column="hr_grade",
    class_values=HR_CLASS_VALUES,
    class_names=HR_CLASS_NAMES,
)


def build_stage2_hr_model(config: Mapping[str, Any] | None = None):
    config = dict(config or {})
    return build_vit_classifier(
        num_classes=HR_TASK_SPEC.num_classes,
        input_shape=tuple(config.get("input_shape", (224, 224, 3))),
        patch_size=int(config.get("patch_size", 16)),
        projection_dim=int(config.get("projection_dim", 64)),
        transformer_layers=int(config.get("transformer_layers", 6)),
        num_heads=int(config.get("num_heads", 4)),
        transformer_units=tuple(config.get("transformer_units", (128, 64))),
        mlp_head_units=tuple(config.get("mlp_head_units", (256, 128))),
        dropout_rate=float(config.get("dropout_rate", 0.1)),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        augmentation=config.get("augmentation"),
        model_name="stage2_hr_vit",
    )


def decode_hr_prediction(probabilities):
    class_index = int(probabilities.argmax())
    grade = HR_TASK_SPEC.decode(class_index)
    return {
        "grade": grade,
        "label": HR_TASK_SPEC.class_name(class_index),
        "confidence": float(probabilities[class_index]),
    }
