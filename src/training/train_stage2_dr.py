from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import evaluate_predictions
from src.eval.plots import plot_confusion_matrix, plot_prediction_examples, plot_training_history
from src.models.stage2_common import (
    build_callbacks,
    build_default_augmentation,
    compute_class_weights,
    ensure_directory,
    filter_task_dataframe,
    history_to_dict,
    load_model,
    load_yaml_config,
    make_dataset,
    resolve_split_dataframe,
    save_json,
)
from src.models.stage2_dr_vit import DR_CLASS_NAMES, DR_TASK_SPEC, build_stage2_dr_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 2 DR Vision Transformer.")
    parser.add_argument("--config", default="configs/model_stage2.yaml")
    parser.add_argument("--master-csv")
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--run-dir")
    parser.add_argument("--figures-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config)
    defaults = config.get("defaults", {})
    paths = config.get("paths", {})
    task = config.get("stage2_dr", {})
    input_shape = task.get("input_shape", defaults.get("input_shape", [224, 224, 3]))

    return {
        "master_csv": args.master_csv or paths.get("master_csv", "data/metadata/master.csv"),
        "train_csv": args.train_csv or paths.get("train_csv", "data/splits/train.csv"),
        "val_csv": args.val_csv or paths.get("val_csv", "data/splits/val.csv"),
        "test_csv": args.test_csv or paths.get("test_csv", "data/splits/test.csv"),
        "run_dir": args.run_dir or task.get("run_dir", "reports/stage2_dr"),
        "figures_dir": args.figures_dir or task.get("figures_dir", "reports/figures/stage2_dr"),
        "epochs": args.epochs or int(defaults.get("epochs", 15)),
        "batch_size": args.batch_size or int(defaults.get("batch_size", 16)),
        "learning_rate": args.learning_rate or float(defaults.get("learning_rate", 3e-4)),
        "seed": args.seed or int(defaults.get("seed", 42)),
        "patience": int(defaults.get("patience", 3)),
        "image_size": tuple(input_shape[:2]),
        "brightness_delta": float(defaults.get("brightness_delta", 0.15)),
        "use_class_weights": bool(defaults.get("use_class_weights", True)),
        "model": task.get("model", {}),
    }


def run_evaluation(
    model: tf.keras.Model,
    split_name: str,
    frame,
    dataset,
    labels: np.ndarray,
    figures_dir: Path,
    run_dir: Path,
) -> None:
    if frame.empty:
        return
    probabilities = model.predict(dataset, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics_payload = evaluate_predictions(labels, predictions, label_names=list(DR_CLASS_NAMES))
    save_json(metrics_payload, run_dir / f"{split_name}_metrics.json")
    plot_confusion_matrix(
        np.asarray(metrics_payload["confusion_matrix"]),
        class_names=list(DR_CLASS_NAMES),
        output_path=figures_dir / f"{split_name}_confusion_matrix.png",
        title=f"Stage 2 DR {split_name.title()} Confusion Matrix",
    )
    plot_prediction_examples(
        image_paths=frame["image_path"].tolist(),
        true_labels=[DR_CLASS_NAMES[index] for index in labels],
        predicted_labels=[DR_CLASS_NAMES[index] for index in predictions],
        confidences=probabilities.max(axis=1).tolist(),
        output_path=figures_dir / f"{split_name}_prediction_examples.png",
        title=f"Stage 2 DR {split_name.title()} Predictions",
    )


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    tf.keras.utils.set_random_seed(settings["seed"])

    run_dir = ensure_directory(settings["run_dir"])
    figures_dir = ensure_directory(settings["figures_dir"])
    checkpoint_path = run_dir / "checkpoints" / "best_model.keras"

    train_frame = filter_task_dataframe(
        resolve_split_dataframe(settings["master_csv"], settings["train_csv"], split_name="train"),
        DR_TASK_SPEC,
    )
    val_frame = filter_task_dataframe(
        resolve_split_dataframe(settings["master_csv"], settings["val_csv"], split_name="val"),
        DR_TASK_SPEC,
    )
    test_frame = filter_task_dataframe(
        resolve_split_dataframe(settings["master_csv"], settings["test_csv"], split_name="test"),
        DR_TASK_SPEC,
    )

    if train_frame.empty:
        raise ValueError("No DR samples were found in the training split.")

    train_dataset, train_labels = make_dataset(
        train_frame,
        DR_TASK_SPEC,
        batch_size=settings["batch_size"],
        image_size=settings["image_size"],
        training=True,
        seed=settings["seed"],
    )
    val_dataset, val_labels = make_dataset(
        val_frame,
        DR_TASK_SPEC,
        batch_size=settings["batch_size"],
        image_size=settings["image_size"],
        seed=settings["seed"],
    )
    test_dataset, test_labels = make_dataset(
        test_frame,
        DR_TASK_SPEC,
        batch_size=settings["batch_size"],
        image_size=settings["image_size"],
        seed=settings["seed"],
    )

    model_config = dict(settings["model"])
    model_config["augmentation"] = build_default_augmentation(settings["brightness_delta"])
    model = build_stage2_dr_model(model_config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(train_labels, DR_TASK_SPEC.num_classes) if settings["use_class_weights"] else None
    fit_kwargs = {
        "x": train_dataset,
        "epochs": settings["epochs"],
        "callbacks": build_callbacks(checkpoint_path, patience=settings["patience"]),
        "class_weight": class_weights,
        "verbose": 1,
    }
    if not val_frame.empty:
        fit_kwargs["validation_data"] = val_dataset

    history = model.fit(**fit_kwargs)

    history_payload = history_to_dict(history)
    save_json(history_payload, run_dir / "history.json")
    save_json(
        {
            "train_samples": int(len(train_frame)),
            "val_samples": int(len(val_frame)),
            "test_samples": int(len(test_frame)),
            "class_weights": class_weights or {},
            "config": settings,
        },
        run_dir / "run_summary.json",
    )
    plot_training_history(history_payload, figures_dir, prefix="stage2_dr")

    best_model = load_model(checkpoint_path) if checkpoint_path.exists() else model
    run_evaluation(best_model, "val", val_frame, val_dataset, val_labels, figures_dir, run_dir)
    if not test_frame.empty:
        run_evaluation(best_model, "test", test_frame, test_dataset, test_labels, figures_dir, run_dir)


if __name__ == "__main__":
    main()

