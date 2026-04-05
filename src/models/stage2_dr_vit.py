from __future__ import annotations

from typing import Any, Mapping

from src.models.stage2_common import Stage2TaskSpec, build_vit_classifier

DR_CLASS_VALUES = (1, 2, 3, 4)
DR_CLASS_NAMES = (
    "Mild NPDR (Grade 1)",
    "Moderate NPDR (Grade 2)",
    "Severe NPDR (Grade 3)",
    "Proliferative DR (Grade 4)",
)

DR_TASK_SPEC = Stage2TaskSpec(
    name="stage2_dr",
    disease_label="dr",
    grade_column="dr_grade",
    class_values=DR_CLASS_VALUES,
    class_names=DR_CLASS_NAMES,
)


def build_stage2_dr_model(config: Mapping[str, Any] | None = None):
    config = dict(config or {})
    return build_vit_classifier(
        num_classes=DR_TASK_SPEC.num_classes,
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
        model_name="stage2_dr_vit",
    )


def decode_dr_prediction(probabilities):
    class_index = int(probabilities.argmax())
    grade = DR_TASK_SPEC.decode(class_index)
    return {
        "grade": grade,
        "label": DR_TASK_SPEC.class_name(class_index),
        "confidence": float(probabilities[class_index]),
    }
