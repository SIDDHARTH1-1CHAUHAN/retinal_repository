from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.models.stage1_tensor_io import (
    expected_stage1_tensor_shape,
    load_stage1_npy_tensor,
)
from src.models.stage1_vit import get_stage1_label_order

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_IMAGE_COLUMN = "image_path"
DEFAULT_LABEL_COLUMN = "disease_label"
KNOWN_IDENTIFIER_COLUMNS = (
    "image_id",
    "sample_id",
    "record_id",
    "case_id",
    "patient_or_case_id",
    "file_name",
    "image_path",
)


@keras.utils.register_keras_serializable(package="stage1_vit")
class RandomBrightness(keras.layers.Layer):
    def __init__(self, factor: float = 0.15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        if not training or self.factor <= 0:
            return inputs
        return tf.image.random_brightness(inputs, max_delta=self.factor)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "factor": self.factor}


def _build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomRotation(0.08, fill_mode="reflect"),
            keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="reflect"),
            RandomBrightness(factor=0.15),
        ],
        name="stage1_augmentation",
    )


@dataclass
class Stage1Datasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    class_weights: dict[int, float]
    label_order: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_csv(path_value: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path_value))


def _normalise_paths(frame: pd.DataFrame, image_column: str) -> pd.DataFrame:
    repo_root = _repo_root()
    resolved = frame.copy()

    def resolve_path(raw_path: Any) -> str:
        value = Path(str(raw_path))
        if value.is_absolute():
            return str(value)
        return str((repo_root / value).resolve())

    resolved[image_column] = resolved[image_column].apply(resolve_path)
    return resolved


def _pick_merge_key(metadata: pd.DataFrame, split: pd.DataFrame) -> str:
    for column in KNOWN_IDENTIFIER_COLUMNS:
        if column in metadata.columns and column in split.columns:
            return column

    common_columns = [
        column
        for column in metadata.columns.intersection(split.columns).tolist()
        if column not in {DEFAULT_LABEL_COLUMN, DEFAULT_IMAGE_COLUMN, "split"}
    ]
    if common_columns:
        return common_columns[0]
    raise ValueError(
        "Unable to merge split CSV with metadata CSV. "
        "Expected a shared identifier column such as image_id, sample_id, "
        "record_id, case_id, patient_or_case_id, file_name, or image_path."
    )


def _merge_split_with_metadata(
    metadata: pd.DataFrame,
    split: pd.DataFrame,
    image_column: str,
    label_column: str,
) -> pd.DataFrame:
    required = {image_column, label_column}
    if required.issubset(split.columns):
        return split.copy()

    merge_key = _pick_merge_key(metadata, split)
    merged = split.merge(
        metadata,
        how="left",
        on=merge_key,
        suffixes=("_split", ""),
        validate="many_to_one",
    )
    if not required.issubset(merged.columns):
        raise ValueError(
            f"Split merge on '{merge_key}' did not produce required columns "
            f"{sorted(required)}. Available columns: {sorted(merged.columns.tolist())}"
        )
    return merged


def _prepare_split_frame(
    split_path: str | Path,
    metadata: pd.DataFrame,
    image_column: str,
    label_column: str,
    label_order: list[str],
) -> pd.DataFrame:
    split = _load_csv(split_path)
    frame = _merge_split_with_metadata(
        metadata=metadata,
        split=split,
        image_column=image_column,
        label_column=label_column,
    )
    frame = _normalise_paths(frame, image_column=image_column)
    frame = frame.dropna(subset=[image_column, label_column]).copy()
    frame[label_column] = frame[label_column].astype(str).str.lower().str.strip()
    invalid_labels = sorted(set(frame[label_column]) - set(label_order))
    if invalid_labels:
        raise ValueError(
            f"Unexpected Stage 1 labels found: {invalid_labels}. "
            f"Expected exactly {label_order}."
        )
    frame["label_index"] = frame[label_column].map(
        {label: index for index, label in enumerate(label_order)}
    )
    missing_files = [
        path_value for path_value in frame[image_column].tolist() if not Path(path_value).exists()
    ]
    if missing_files:
        preview = missing_files[:5]
        raise FileNotFoundError(
            f"{len(missing_files)} image paths from {split_path} do not exist. "
            f"Examples: {preview}"
        )
    return frame.reset_index(drop=True)


def _read_standard_image_file(
    image_path: tf.Tensor,
    image_size: tuple[int, int],
) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.ensure_shape(image, expected_stage1_tensor_shape(image_size))
    return tf.clip_by_value(image, 0.0, 1.0)


def _read_numpy_tensor_file(
    image_path: tf.Tensor,
    image_size: tuple[int, int],
) -> tf.Tensor:
    tensor = tf.numpy_function(
        func=lambda path: load_stage1_npy_tensor(path, image_size=image_size),
        inp=[image_path],
        Tout=tf.float32,
    )
    return tf.ensure_shape(tensor, expected_stage1_tensor_shape(image_size))


def _read_image(image_path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    lower_path = tf.strings.lower(image_path)
    is_numpy_tensor = tf.strings.regex_full_match(lower_path, r".*\.npy")
    image = tf.cond(
        is_numpy_tensor,
        lambda: _read_numpy_tensor_file(image_path=image_path, image_size=image_size),
        lambda: _read_standard_image_file(image_path=image_path, image_size=image_size),
    )
    image = tf.cast(image, tf.float32)
    image = tf.ensure_shape(image, expected_stage1_tensor_shape(image_size))
    return tf.clip_by_value(image, 0.0, 1.0)


def _encode_example(
    image_path: tf.Tensor,
    label_index: tf.Tensor,
    image_size: tuple[int, int],
    num_classes: int,
    augmentation: keras.Sequential | None = None,
    sample_weight: tf.Tensor | None = None,
    training: bool = False,
):
    image = _read_image(image_path=image_path, image_size=image_size)
    if augmentation is not None:
        image = augmentation(image, training=training)
        image = tf.clip_by_value(image, 0.0, 1.0)
    label = tf.one_hot(tf.cast(label_index, tf.int32), depth=num_classes, dtype=tf.float32)
    if sample_weight is None:
        return image, label
    return image, label, sample_weight


def _build_dataset(
    frame: pd.DataFrame,
    image_column: str,
    batch_size: int,
    image_size: tuple[int, int],
    num_classes: int,
    training: bool,
    sample_weight_map: dict[int, float] | None = None,
    seed: int = 42,
) -> tf.data.Dataset:
    paths = frame[image_column].astype(str).tolist()
    labels = frame["label_index"].astype("int32").tolist()
    if sample_weight_map:
        weights = [float(sample_weight_map[int(label)]) for label in labels]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=max(len(frame), 1), seed=seed, reshuffle_each_iteration=True)

    augmentation = _build_augmentation() if training else None
    if sample_weight_map:
        dataset = dataset.map(
            lambda path, label, weight: _encode_example(
                image_path=path,
                label_index=label,
                image_size=image_size,
                num_classes=num_classes,
                augmentation=augmentation,
                sample_weight=weight,
                training=training,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda path, label: _encode_example(
                image_path=path,
                label_index=label,
                image_size=image_size,
                num_classes=num_classes,
                augmentation=augmentation,
                training=training,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def compute_class_weights(frame: pd.DataFrame, label_order: list[str]) -> dict[int, float]:
    counts = frame["label_index"].value_counts().sort_index()
    total = float(len(frame))
    num_classes = len(label_order)
    weights: dict[int, float] = {}
    for index in range(num_classes):
        count = float(counts.get(index, 0.0))
        weights[index] = 0.0 if count == 0 else total / (num_classes * count)
    return weights


def load_stage1_datasets(config: dict[str, Any]) -> Stage1Datasets:
    data_config = config["data"]
    training_config = config["training"]
    label_order = data_config.get("label_order", get_stage1_label_order())
    image_column = data_config.get("image_column", DEFAULT_IMAGE_COLUMN)
    label_column = data_config.get("label_column", DEFAULT_LABEL_COLUMN)
    image_size = tuple(int(value) for value in data_config.get("input_size", [224, 224]))
    batch_size = int(training_config.get("batch_size", 16))
    seed = int(training_config.get("seed", 42))

    metadata = _load_csv(data_config["metadata_path"])
    train_frame = _prepare_split_frame(
        split_path=data_config["train_split_path"],
        metadata=metadata,
        image_column=image_column,
        label_column=label_column,
        label_order=label_order,
    )
    val_frame = _prepare_split_frame(
        split_path=data_config["val_split_path"],
        metadata=metadata,
        image_column=image_column,
        label_column=label_column,
        label_order=label_order,
    )
    test_frame = _prepare_split_frame(
        split_path=data_config["test_split_path"],
        metadata=metadata,
        image_column=image_column,
        label_column=label_column,
        label_order=label_order,
    )

    class_weights = (
        compute_class_weights(train_frame, label_order)
        if training_config.get("use_class_weights", True)
        else {}
    )

    train_dataset = _build_dataset(
        frame=train_frame,
        image_column=image_column,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=len(label_order),
        training=True,
        sample_weight_map=class_weights if class_weights else None,
        seed=seed,
    )
    val_dataset = _build_dataset(
        frame=val_frame,
        image_column=image_column,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=len(label_order),
        training=False,
        seed=seed,
    )
    test_dataset = _build_dataset(
        frame=test_frame,
        image_column=image_column,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=len(label_order),
        training=False,
        seed=seed,
    )

    return Stage1Datasets(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        class_weights=class_weights,
        label_order=label_order,
    )
