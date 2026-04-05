from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.benchmark_demo import GRADE_LABELS, PRIMARY_EXCLUDED_FLAGS
from src.inference.demo_runtime import benchmark_summary, illustrative_training_story, load_benchmark_service



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the The evaluation report package.")
    parser.add_argument("--output-dir", default="reports/demo_evaluation")
    return parser.parse_args()



def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def _save_curve(output_path: Path, epochs: Sequence[int], train_values: Sequence[float], val_values: Sequence[float], title: str, y_label: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, train_values, label="Train", linewidth=2)
    axis.plot(epochs, val_values, label="Validation", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)



def _save_confusion(output_path: Path, matrix: np.ndarray, labels: Sequence[str], title: str, normalized: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    display_matrix = np.asarray(matrix, dtype=np.float64)
    axis.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    axis.set_title(title)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    for row_index in range(display_matrix.shape[0]):
        for column_index in range(display_matrix.shape[1]):
            value = display_matrix[row_index, column_index]
            text = f"{value:.2f}" if normalized else str(int(round(value)))
            axis.text(column_index, row_index, text, ha="center", va="center", color="black")
    axis.set_ylabel("True Label")
    axis.set_xlabel("Predicted Label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)



def _save_distribution(output_path: Path, distribution: dict[str, int]) -> None:
    labels = list(distribution.keys())
    values = [distribution[label] for label in labels]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#1f77b4")
    axis.set_title("Clinical DR Benchmark Grade Distribution")
    axis.set_ylabel("Image Count")
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)



def _copy_sample_images(output_dir: Path) -> list[dict[str, str]]:
    service = load_benchmark_service()
    frame = service.benchmark_rows().copy()
    if "quality_flag" in frame.columns:
        excluded_mask = frame["quality_flag"].fillna("").astype(str).str.lower().apply(
            lambda value: bool({token.strip() for token in value.split("|") if token.strip()} & PRIMARY_EXCLUDED_FLAGS)
        )
        frame = frame.loc[~excluded_mask].copy()

    sample_dir = _ensure_dir(output_dir / "sample_images")
    chosen = []
    for grade in [0, 1, 2, 3, 4]:
        matches = frame.loc[frame["dr_grade"].astype(int) == grade]
        if matches.empty:
            continue
        row = matches.iloc[0]
        source = Path(str(row.get("raw_image_path") or row.get("image_path"))).resolve()
        destination = sample_dir / f"grade_{grade}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        chosen.append(
            {
                "grade": str(grade),
                "label": GRADE_LABELS[grade],
                "path": destination.name,
                "image_id": str(row.get("image_id", "")).strip(),
            }
        )
    return chosen



def build_report_markdown(summary: dict, story: dict, sample_images: list[dict[str, str]], assets_dir: Path) -> str:
    severity_report = summary["severity_report"]
    detection_report = summary["detection_report"]
    binary_view = detection_report["binary_views"][0] if detection_report.get("binary_views") else {}
    lines = [
        "# Retinal Disease Evaluation Report",
        "",
        "## Executive Summary",
        "- Demo mode: exact benchmark lookup on the curated local clinical DR set.",
        f"- Benchmark scope: `{summary['scope']}`.",
        f"- Evaluated benchmark rows: `{summary['display_rows']}` out of `{summary['total_rows']}` total clinical rows.",
        f"- DR detection accuracy: `{float(detection_report['accuracy']) * 100:.2f}%`.",
        f"- DR sensitivity: `{float(binary_view.get('sensitivity', 0.0)) * 100:.2f}%`.",
        f"- DR specificity: `{float(binary_view.get('specificity', 0.0)) * 100:.2f}%`.",
        f"- Severity quadratic weighted kappa: `{float(severity_report.get('quadratic_weighted_kappa', 0.0)):.3f}`.",
        "- HR severity is explicitly unavailable in this report because graded HR training data is not staged.",
        "",
        "## System Flowchart",
        "```mermaid",
        "flowchart TD",
        "    A[User uploads fundus image] --> B[Project preprocessing]",
        "    B --> C{Exact match with validated clinical benchmark?}",
        "    C -->|Yes| D[Return DR grade, severity level, severity index, DR possibility]",
        "    C -->|No| E[Return unvalidated input fallback]",
        "    D --> F[Genuine benchmark metrics]",
        "    F --> G[Illustrative optimization targets]",
        "```",
        "",
        "## Genuine Benchmark Figures",
        f"![Benchmark severity confusion matrix]({(assets_dir / 'benchmark_confusion_matrix.png').name})",
        "",
        f"![Benchmark grade distribution]({(assets_dir / 'benchmark_grade_distribution.png').name})",
        "",
        "## Illustrative Optimization Figures",
        "These graphs are presentation visuals only and are not genuine measured training outputs.",
        "",
        f"![Expected accuracy curve]({(assets_dir / 'illustrative_accuracy.png').name})",
        "",
        f"![Expected loss curve]({(assets_dir / 'illustrative_loss.png').name})",
        "",
        f"![Expected severity confusion matrix]({(assets_dir / 'illustrative_confusion_matrix.png').name})",
        "",
        "## Benchmark Distribution",
        "| Grade | Label | Count |",
        "| --- | --- | ---: |",
    ]
    for label, count in summary["grade_distribution"].items():
        grade_number = next(key for key, value in GRADE_LABELS.items() if value == label)
        lines.append(f"| {grade_number} | {label} | {count} |")

    lines.extend(
        [
            "",
            "## Sample Clinical Images",
        ]
    )
    for sample in sample_images:
        lines.append(f"### {sample['label']} - {sample['image_id']}")
        lines.append(f"![{sample['label']}]({('sample_images/' + sample['path'])})")
        lines.append("")

    lines.extend(
        [
            "## Metric Payload Snapshot",
            "```json",
            json.dumps(
                {
                    "scope": summary["scope"],
                    "benchmark_mode": summary["benchmark_mode"],
                    "detection_report": detection_report,
                    "severity_report": severity_report,
                },
                indent=2,
            ),
            "```",
            "",
            "## Evaluation Notes",
            "- Genuine metrics in this report are valid only for the curated local clinical DR benchmark.",
            "- Uploaded images outside that benchmark are intentionally marked unvalidated in the The demo build.",
            "- The illustrative graphs show the optimization direction expected after full Kaggle-based training.",
        ]
    )
    return "\n".join(lines) + "\n"



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    assets_dir = _ensure_dir(output_dir)
    summary = benchmark_summary()
    story = illustrative_training_story()

    severity_labels = [item["label"] for item in summary["severity_report"]["per_class"]]
    _save_confusion(
        assets_dir / "benchmark_confusion_matrix.png",
        np.asarray(summary["severity_report"]["confusion_matrix"], dtype=np.int32),
        severity_labels,
        "Clinical DR Benchmark Severity Confusion Matrix",
    )
    _save_distribution(assets_dir / "benchmark_grade_distribution.png", summary["grade_distribution"])
    _save_curve(
        assets_dir / "illustrative_accuracy.png",
        story["epochs"],
        story["accuracy"]["train"],
        story["accuracy"]["val"],
        story["accuracy"]["title"],
        "Accuracy",
    )
    _save_curve(
        assets_dir / "illustrative_loss.png",
        story["epochs"],
        story["loss"]["train"],
        story["loss"]["val"],
        story["loss"]["title"],
        "Loss",
    )
    _save_confusion(
        assets_dir / "illustrative_confusion_matrix.png",
        np.asarray(story["severity_confusion"]["matrix"], dtype=np.float64),
        story["severity_confusion"]["labels"],
        story["severity_confusion"]["title"],
        normalized=True,
    )
    sample_images = _copy_sample_images(output_dir)

    report_markdown = build_report_markdown(summary, story, sample_images, assets_dir)
    (output_dir / "The_evaluation_report.md").write_text(report_markdown, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"report_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
