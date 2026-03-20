from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
REQUIRED_COLUMNS = (
    "image_id",
    "image_path",
    "disease_label",
    "dr_grade",
    "hr_grade",
)


@dataclass(frozen=True)
class Stage2TaskSpec:
    name: str
    disease_label: str
    grade_column: str
    class_values: tuple[int, ...]
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_values)

    def encode(self, grade: int | float | str) -> int:
        return self.class_values.index(int(grade))

    def decode(self, class_index: int) -> int:
        return self.class_values[int(class_index)]

    def class_name(self, class_index: int) -> str:
        return self.class_names[int(class_index)]


class BrightnessJitter(tf.keras.layers.Layer):
    def __init__(self, delta: float = 0.15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delta = float(delta)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if training:
            inputs = tf.image.random_brightness(inputs, max_delta=self.delta)
        return tf.clip_by_value(inputs, 0.0, 1.0)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"delta": self.delta})
        return config


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)

    def call(self, images: tf.Tensor) -> tf.Tensor:
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        batch_size = tf.shape(images)[0]
        patch_dims = tf.shape(patches)[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_patches = int(num_patches)
        self.projection_dim = int(projection_dim)
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim,
        )

    def call(self, patch: tf.Tensor) -> tf.Tensor:
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def mlp(
    inputs: tf.Tensor,
    hidden_units: Sequence[int],
    dropout_rate: float,
) -> tf.Tensor:
    features = inputs
    for units in hidden_units:
        features = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    return features


def build_default_augmentation(brightness_delta: float = 0.15) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.08),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            BrightnessJitter(delta=brightness_delta),
        ],
        name="stage2_augmentation",
    )


def build_vit_classifier(
    num_classes: int,
    input_shape: Sequence[int] = (224, 224, 3),
    patch_size: int = 16,
    projection_dim: int = 64,
    transformer_layers: int = 6,
    num_heads: int = 4,
    transformer_units: Sequence[int] = (128, 64),
    mlp_head_units: Sequence[int] = (256, 128),
    dropout_rate: float = 0.1,
    attention_dropout: float = 0.1,
    augmentation: tf.keras.Model | None = None,
    model_name: str = "stage2_vit_classifier",
) -> tf.keras.Model:
    height, width, _ = input_shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("input_shape must be divisible by patch_size.")

    num_patches = (height // patch_size) * (width // patch_size)
    inputs = tf.keras.Input(shape=input_shape, name="image")
    encoded = augmentation(inputs) if augmentation is not None else inputs
    encoded = Patches(patch_size=patch_size, name="patches")(encoded)
    encoded = PatchEncoder(
        num_patches=num_patches,
        projection_dim=projection_dim,
        name="patch_encoder",
    )(encoded)

    for layer_index in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln_1_{layer_index}")(
            encoded
        )
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=attention_dropout,
            name=f"mha_{layer_index}",
        )(x1, x1)
        x2 = tf.keras.layers.Add(name=f"skip_attention_{layer_index}")(
            [attention_output, encoded]
        )
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln_2_{layer_index}")(
            x2
        )
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        encoded = tf.keras.layers.Add(name=f"skip_mlp_{layer_index}")([x3, x2])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="representation_ln")(
        encoded
    )
    representation = tf.keras.layers.GlobalAveragePooling1D(name="representation_pool")(
        representation
    )
    representation = tf.keras.layers.Dropout(dropout_rate, name="representation_dropout")(
        representation
    )
    features = mlp(
        representation,
        hidden_units=mlp_head_units,
        dropout_rate=dropout_rate,
    )
    logits = tf.keras.layers.Dense(num_classes, activation="softmax", name="severity")(features)
    return tf.keras.Model(inputs=inputs, outputs=logits, name=model_name)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _coerce_frame_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str)
    frame["disease_label"] = frame["disease_label"].astype(str).str.lower()
    for column in ("dr_grade", "hr_grade"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "split" in frame.columns:
        frame["split"] = frame["split"].astype(str).str.lower()
    return frame


def resolve_split_dataframe(
    master_csv: str | Path,
    split_csv: str | Path,
    split_name: str | None = None,
) -> pd.DataFrame:
    split_path = Path(split_csv)
    master_path = Path(master_csv)

    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")

    split_frame = pd.read_csv(split_path)
    master_frame = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in split_frame.columns]
    if missing_columns:
        if master_frame.empty or "image_id" not in split_frame.columns:
            raise ValueError(
                f"Split file {split_path} is missing columns {missing_columns} and cannot be merged."
            )
        merge_columns = ["image_id", *missing_columns]
        if "split" not in split_frame.columns and "split" in master_frame.columns:
            merge_columns.append("split")
        split_frame = split_frame.merge(master_frame[merge_columns], on="image_id", how="left")

    if split_name and "split" not in split_frame.columns:
        split_frame["split"] = split_name

    missing_after_merge = [column for column in REQUIRED_COLUMNS if column not in split_frame.columns]
    if missing_after_merge:
        raise ValueError(f"Missing required columns after merge: {missing_after_merge}")

    split_frame = _coerce_frame_types(split_frame)
    split_frame = split_frame.dropna(subset=["image_path", "disease_label"])
    split_frame = split_frame.drop_duplicates(subset=["image_id"])
    return split_frame


def filter_task_dataframe(frame: pd.DataFrame, spec: Stage2TaskSpec) -> pd.DataFrame:
    filtered = frame.copy()
    filtered = filtered[filtered["disease_label"] == spec.disease_label]
    filtered = filtered[filtered[spec.grade_column].notna()]
    filtered[spec.grade_column] = filtered[spec.grade_column].astype(int)
    filtered = filtered[filtered[spec.grade_column].isin(spec.class_values)]
    return filtered.reset_index(drop=True)


def encode_labels(frame: pd.DataFrame, spec: Stage2TaskSpec) -> np.ndarray:
    return np.asarray([spec.encode(value) for value in frame[spec.grade_column].tolist()], dtype=np.int32)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    weights: dict[int, float] = {}
    if labels.size == 0:
        return weights
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    for label, count in zip(unique, counts):
        weights[int(label)] = float(total / (len(unique) * count))
    for label in range(num_classes):
        weights.setdefault(label, 1.0)
    return weights


def _load_numpy_tensor(path: str, image_size: Sequence[int]) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    array = np.asarray(array, dtype=np.float32)
    expected_shape = (int(image_size[0]), int(image_size[1]), 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"Expected a preprocessed tensor with shape {expected_shape}, received {array.shape} from {path}."
        )
    min_value = float(array.min())
    max_value = float(array.max())
    if min_value < 0.0 or max_value > 1.0:
        raise ValueError(f"Expected normalized tensor values in [0, 1], received range [{min_value}, {max_value}].")
    return array


def _load_standard_image(path: str, image_size: Sequence[int]) -> np.ndarray:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image.numpy()


def _decode_path_value(path_value: Any) -> str:
    if isinstance(path_value, bytes):
        return path_value.decode("utf-8")
    if isinstance(path_value, np.bytes_):
        return path_value.decode("utf-8")
    return str(path_value)


def _load_tensor_numpy(path_bytes: Any, image_size: Sequence[int]) -> np.ndarray:
    path = Path(_decode_path_value(path_bytes))
    if path.suffix.lower() == ".npy":
        return _load_numpy_tensor(str(path), image_size)
    return _load_standard_image(str(path), image_size)


def load_tensor_from_path(path: tf.Tensor, image_size: Sequence[int] = (224, 224)) -> tf.Tensor:
    tensor = tf.numpy_function(
        func=lambda value: _load_tensor_numpy(value, image_size),
        inp=[path],
        Tout=tf.float32,
    )
    tensor.set_shape([int(image_size[0]), int(image_size[1]), 3])
    return tensor


def make_dataset(
    frame: pd.DataFrame,
    spec: Stage2TaskSpec,
    batch_size: int,
    image_size: Sequence[int] = (224, 224),
    training: bool = False,
    seed: int = 42,
) -> tuple[tf.data.Dataset, np.ndarray]:
    labels = encode_labels(frame, spec)
    one_hot = tf.one_hot(labels, depth=spec.num_classes)
    dataset = tf.data.Dataset.from_tensor_slices((frame["image_path"].tolist(), one_hot))

    if training and len(frame) > 0:
        dataset = dataset.shuffle(
            buffer_size=max(len(frame), batch_size),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, target: (load_tensor_from_path(path, image_size=image_size), target),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset, labels


def default_converter(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    ensure_directory(output.parent)
    output.write_text(json.dumps(payload, indent=2, default=default_converter), encoding="utf-8")


def history_to_dict(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def build_callbacks(
    checkpoint_path: str | Path,
    patience: int = 3,
    monitor: str = "val_loss",
) -> list[tf.keras.callbacks.Callback]:
    checkpoint = Path(checkpoint_path)
    ensure_directory(checkpoint.parent)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def get_custom_objects() -> dict[str, Any]:
    return {
        "BrightnessJitter": BrightnessJitter,
        "Patches": Patches,
        "PatchEncoder": PatchEncoder,
    }


def load_model(model_path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError:
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}
