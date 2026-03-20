from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.contracts import MASTER_METADATA_PATH, load_yaml_config, resolve_repo_path, save_metadata


def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for preprocessing. Install opencv-python before running this script."
        ) from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess retinal images into normalized 224x224 tensors.")
    parser.add_argument("--config", default="configs/data/data_config.yaml", help="Path to the data config.")
    parser.add_argument("--metadata", default=str(MASTER_METADATA_PATH), help="Input metadata CSV path.")
    parser.add_argument(
        "--output-metadata",
        default=str(MASTER_METADATA_PATH),
        help="Output metadata CSV path. Defaults to updating master.csv in place.",
    )
    parser.add_argument("--output-root", default="data/processed", help="Directory for normalized .npy tensors.")
    return parser.parse_args()


def crop_black_borders(image_rgb: np.ndarray, threshold: int) -> np.ndarray:
    cv2 = require_cv2()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero((gray > threshold).astype(np.uint8))
    if coords is None:
        return image_rgb
    x, y, width, height = cv2.boundingRect(coords)
    return image_rgb[y : y + height, x : x + width]


def apply_clahe(image_rgb: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    cv2 = require_cv2()
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_image_array(image_rgb: np.ndarray, settings: dict) -> np.ndarray:
    cv2 = require_cv2()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape (H, W, 3).")
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)

    crop_settings = settings.get("crop_black_borders", {})
    if crop_settings.get("enabled", True):
        image_rgb = crop_black_borders(image_rgb, threshold=int(crop_settings.get("threshold", 7)))

    blur_settings = settings.get("gaussian_blur", {})
    if blur_settings.get("enabled", True):
        kernel = tuple(blur_settings.get("kernel_size", [5, 5]))
        image_rgb = cv2.GaussianBlur(image_rgb, kernel, sigmaX=float(blur_settings.get("sigma", 0.0)))

    clahe_settings = settings.get("clahe", {})
    if clahe_settings.get("enabled", True):
        image_rgb = apply_clahe(
            image_rgb,
            clip_limit=float(clahe_settings.get("clip_limit", 2.0)),
            tile_grid_size=tuple(clahe_settings.get("tile_grid_size", [8, 8])),
        )

    resize_settings = settings.get("resize", {})
    width, height = resize_settings.get("size", [224, 224])
    image_rgb = cv2.resize(image_rgb, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return image_rgb.astype(np.float32) / float(settings.get("normalize", {}).get("scale", 255.0))


def preprocess_image(raw_path: Path, settings: dict) -> np.ndarray:
    cv2 = require_cv2()
    bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {raw_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return preprocess_image_array(image_rgb, settings)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(resolve_repo_path(project_root, args.config))
    metadata_path = resolve_repo_path(project_root, args.metadata)
    output_metadata_path = resolve_repo_path(project_root, args.output_metadata)
    output_root = resolve_repo_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(metadata_path)
    preprocess_settings = config.get("preprocessing", {})

    if "raw_image_path" not in frame.columns:
        frame["raw_image_path"] = frame["image_path"]

    for index, row in frame.iterrows():
        raw_path = Path(row["raw_image_path"])
        tensor = preprocess_image(raw_path, preprocess_settings)
        dataset_dir = output_root / str(row["source_dataset"]).lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = dataset_dir / f"{row['image_id']}.npy"
        np.save(tensor_path, tensor.astype(np.float32))
        frame.at[index, "image_path"] = str(tensor_path.resolve())

    save_metadata(frame, output_metadata_path)
    print(f"Preprocessed {len(frame)} images into {output_root}")


if __name__ == "__main__":
    main()

