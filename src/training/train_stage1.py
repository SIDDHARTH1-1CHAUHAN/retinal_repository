from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.stage1_vit import build_stage1_vit, compile_stage1_model
from src.training.datasets_stage1 import Stage1Datasets, load_stage1_datasets


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Stage 1 ViT model for normal vs DR vs HR classification."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs" / "model_stage1.yaml",
        help="Path to the Stage 1 YAML config.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must parse to a dictionary.")
    return config


def _resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    repo_root = _repo_root()

    for section_name in ("data", "outputs"):
        section = resolved.get(section_name, {})
        for key, value in section.items():
            if not isinstance(value, str):
                continue
            if not any(token in key for token in ("path", "dir", "root")):
                continue
            candidate = Path(value)
            section[key] = str(candidate if candidate.is_absolute() else (repo_root / candidate))
    return resolved


def _ensure_output_directories(output_config: dict[str, Any]) -> dict[str, Path]:
    run_root = Path(output_config["run_root"])
    figures_dir = Path(output_config["figures_dir"])
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    for directory in (run_root, figures_dir, checkpoints_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "figures_dir": figures_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
    }


def _metrics_from_confusion_matrix(confusion: np.ndarray) -> dict[str, float]:
    total = confusion.sum()
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1_score = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1_score": float(f1_score.mean()),
    }


def _save_confusion_matrix(
    confusion: np.ndarray,
    label_order: list[str],
    csv_path: Path,
    figure_path: Path,
) -> None:
    frame = pd.DataFrame(confusion, index=label_order, columns=label_order)
    frame.to_csv(csv_path, index=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(6, 5))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Stage 1 Confusion Matrix")
    plt.colorbar()
    tick_positions = np.arange(len(label_order))
    plt.xticks(tick_positions, label_order)
    plt.yticks(tick_positions, label_order)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            value = int(confusion[row_index, column_index])
            color = "white" if value > threshold else "black"
            plt.text(column_index, row_index, value, ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def _predict_labels(model: keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    probabilities = model.predict(dataset, verbose=0)
    return np.argmax(probabilities, axis=1)


def _labels_from_frame(frame: pd.DataFrame) -> np.ndarray:
    return frame["label_index"].astype("int32").to_numpy()


def _select_eval_split(
    datasets: Stage1Datasets, split_name: str
) -> tuple[str, tf.data.Dataset, pd.DataFrame]:
    split_name = split_name.lower().strip()
    if split_name == "val":
        return "val", datasets.val, datasets.val_frame
    if split_name == "test":
        return "test", datasets.test, datasets.test_frame
    raise ValueError("outputs.evaluation_split must be either 'val' or 'test'.")


def _build_model(config: dict[str, Any], num_classes: int) -> keras.Model:
    data_config = config["data"]
    model_config = config["model"]
    image_height, image_width = (int(value) for value in data_config["input_size"])
    model = build_stage1_vit(
        input_shape=(image_height, image_width, 3),
        image_size=(image_height, image_width),
        patch_size=int(model_config.get("patch_size", 16)),
        hidden_size=int(model_config.get("hidden_size", 768)),
        transformer_layers=int(model_config.get("transformer_layers", 12)),
        num_heads=int(model_config.get("num_heads", 12)),
        mlp_dim=int(model_config.get("mlp_dim", 3072)),
        dropout_rate=float(model_config.get("dropout_rate", 0.1)),
        attention_dropout_rate=float(model_config.get("attention_dropout_rate", 0.1)),
        num_classes=num_classes,
        classifier=str(model_config.get("classifier", "token")),
        model_name=str(model_config.get("name", "stage1_vit_b16")),
    )
    return compile_stage1_model(
        model=model,
        learning_rate=float(config["training"].get("learning_rate", 1e-4)),
        num_classes=num_classes,
    )


def _build_callbacks(config: dict[str, Any], output_paths: dict[str, Path]) -> list[keras.callbacks.Callback]:
    training_config = config["training"]
    output_config = config["outputs"]
    best_model_path = output_paths["run_root"] / output_config["best_model_name"]
    checkpoint_pattern = output_paths["checkpoints_dir"] / "epoch-{epoch:02d}-val-f1-{val_f1_score:.4f}.weights.h5"
    history_path = output_paths["logs_dir"] / output_config["history_name"]

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_f1_score",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_pattern),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_f1_score",
            mode="max",
            patience=int(training_config.get("early_stopping_patience", 3)),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(history_path), append=False),
    ]


def _save_training_contract(
    output_paths: dict[str, Path],
    output_config: dict[str, Any],
    label_order: list[str],
    input_size: list[int],
) -> None:
    label_map_path = output_paths["run_root"] / output_config["label_map_name"]
    payload = {
        "labels": label_order,
        "input_shape": [int(input_size[0]), int(input_size[1]), 3],
        "prediction_fields": ["disease", "severity", "confidence"],
        "saved_model_format": "keras_v3",
    }
    label_map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    raw_config = _load_config(args.config)
    config = _resolve_config_paths(raw_config)

    seed = int(config["training"].get("seed", 42))
    tf.keras.utils.set_random_seed(seed)

    output_paths = _ensure_output_directories(config["outputs"])
    datasets = load_stage1_datasets(config)
    model = _build_model(config, num_classes=len(datasets.label_order))

    callbacks = _build_callbacks(config, output_paths)
    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=int(config["training"].get("epochs", 15)),
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = output_paths["run_root"] / config["outputs"]["final_model_name"]
    model.save(final_model_path)

    eval_split_name, eval_dataset, eval_frame = _select_eval_split(
        datasets=datasets,
        split_name=str(config["outputs"].get("evaluation_split", "val")),
    )
    eval_metrics = model.evaluate(eval_dataset, return_dict=True, verbose=0)
    y_true = _labels_from_frame(eval_frame)
    y_pred = _predict_labels(model, eval_dataset)
    confusion = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(datasets.label_order),
    ).numpy()
    aggregate_metrics = _metrics_from_confusion_matrix(confusion)

    output_config = config["outputs"]
    confusion_csv_path = output_paths["run_root"] / output_config["confusion_matrix_name"]
    confusion_figure_path = output_paths["figures_dir"] / output_config["confusion_matrix_figure_name"]
    _save_confusion_matrix(
        confusion=confusion,
        label_order=datasets.label_order,
        csv_path=confusion_csv_path,
        figure_path=confusion_figure_path,
    )
    _save_training_contract(
        output_paths=output_paths,
        output_config=output_config,
        label_order=datasets.label_order,
        input_size=config["data"]["input_size"],
    )

    evaluation_path = output_paths["run_root"] / output_config["evaluation_name"]
    evaluation_payload = {
        "evaluation_split": eval_split_name,
        "label_order": datasets.label_order,
        "best_model_path": str(output_paths["run_root"] / output_config["best_model_name"]),
        "final_model_path": str(final_model_path),
        "train_samples": int(len(datasets.train_frame)),
        "val_samples": int(len(datasets.val_frame)),
        "test_samples": int(len(datasets.test_frame)),
        "class_weights": datasets.class_weights,
        "history_keys": list(history.history.keys()),
        "keras_metrics": {key: float(value) for key, value in eval_metrics.items()},
        "aggregate_metrics": aggregate_metrics,
        "confusion_matrix": confusion.tolist(),
    }
    evaluation_path.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")

    print(json.dumps(evaluation_payload, indent=2))


if __name__ == "__main__":
    main()
