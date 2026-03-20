from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess_images import preprocess_image_array, require_cv2
from src.data.contracts import load_yaml_config
from src.inference.predict_pipeline import RetinalDiseasePredictor
from src.models.stage1_vit import get_stage1_custom_objects


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_preprocessing_settings(config_path: str | Path = "configs/data/data_config.yaml") -> dict[str, Any]:
    root = repo_root()
    candidate = Path(config_path)
    resolved = candidate if candidate.is_absolute() else root / candidate
    config = load_yaml_config(resolved)
    return config.get("preprocessing", {})


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    cv2 = require_cv2()
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode the uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_uploaded_image(
    image_bytes: bytes,
    config_path: str | Path = "configs/data/data_config.yaml",
) -> tuple[np.ndarray, np.ndarray]:
    image_rgb = decode_uploaded_image(image_bytes)
    settings = load_preprocessing_settings(config_path)
    tensor = preprocess_image_array(image_rgb, settings)
    return image_rgb, tensor


def load_predictor(config_path: str | Path = "configs/model_stage2.yaml") -> RetinalDiseasePredictor:
    root = repo_root()
    candidate = Path(config_path)
    resolved = candidate if candidate.is_absolute() else root / candidate
    return RetinalDiseasePredictor.from_config(
        config_path=resolved,
        stage1_custom_objects=get_stage1_custom_objects(),
    )


def model_availability(config_path: str | Path = "configs/model_stage2.yaml") -> dict[str, Path]:
    root = repo_root()
    candidate = Path(config_path)
    resolved = candidate if candidate.is_absolute() else root / candidate
    config = load_yaml_config(resolved)
    inference = config.get("inference", {})
    stage1_cfg = inference.get("stage1", {})
    stage2_cfg = inference.get("stage2", {})
    return {
        "stage1": root / stage1_cfg.get("model_path", ""),
        "stage2_dr": root / stage2_cfg.get("dr_model_path", ""),
        "stage2_hr": root / stage2_cfg.get("hr_model_path", ""),
    }
