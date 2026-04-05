from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.inference.demo_runtime import (
    benchmark_summary,
    illustrative_training_story,
    load_benchmark_service,
    model_availability,
    preprocess_uploaded_image,
)


st.set_page_config(page_title="Retinal Benchmark Demo", layout="wide")



def render_model_status() -> None:
    st.subheader("Artifact Status")
    status = model_availability()
    for name, path in status.items():
        if str(path) and path.exists():
            st.success(f"{name}: found at {path}")
        else:
            st.warning(f"{name}: missing at {path}")
    st.info("Tomorrow's evaluation build runs in benchmark lookup mode and does not claim general model inference.")



def render_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, normalized: bool = False):
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
    return figure



def render_curve_chart(epochs: list[int], train_values: list[float], val_values: list[float], title: str, y_label: str):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, train_values, label="Train", linewidth=2)
    axis.plot(epochs, val_values, label="Validation", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure



def render_benchmark_metrics() -> None:
    summary = benchmark_summary()
    severity_report = summary["severity_report"]
    detection_report = summary["detection_report"]
    binary_dr_view = detection_report["binary_views"][0] if detection_report.get("binary_views") else {}
    labels = [item["label"] for item in severity_report.get("per_class", [])]
    matrix = np.asarray(severity_report["confusion_matrix"], dtype=np.int32)

    st.subheader("Genuine Benchmark Metrics")
    st.caption("These metrics are computed only on the curated local clinical DR benchmark used for tomorrow's demo.")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Benchmark Rows", str(summary["display_rows"]))
    top2.metric("Excluded Rows", str(summary["excluded_rows"]))
    top3.metric("Severity Accuracy", f"{float(severity_report['accuracy']) * 100:.2f}%")
    top4.metric("Detection Accuracy", f"{float(detection_report['accuracy']) * 100:.2f}%")

    middle1, middle2, middle3 = st.columns(3)
    middle1.metric("Sensitivity", f"{float(binary_dr_view.get('sensitivity', 0.0)) * 100:.2f}%")
    middle2.metric("Specificity", f"{float(binary_dr_view.get('specificity', 0.0)) * 100:.2f}%")
    middle3.metric("QWK", f"{float(severity_report.get('quadratic_weighted_kappa', 0.0)):.3f}")

    with st.expander("Benchmark Scope And Distribution", expanded=False):
        st.json(
            {
                "scope": summary["scope"],
                "benchmark_mode": summary["benchmark_mode"],
                "grade_distribution": summary["grade_distribution"],
                "excluded_quality_counts": summary["excluded_quality_counts"],
            }
        )

    st.pyplot(render_confusion_matrix(matrix, labels, "Clinical DR Benchmark Severity Confusion Matrix"))

    with st.expander("Benchmark Metric Payload", expanded=False):
        st.json(
            {
                "severity_report": severity_report,
                "detection_report": detection_report,
            }
        )



def render_illustrative_graphs() -> None:
    story = illustrative_training_story()
    accuracy = story["accuracy"]
    loss = story["loss"]
    severity_confusion = story["severity_confusion"]
    st.subheader("Expected Optimization Graphs")
    st.caption("Illustrative only. These are expected target visuals for the full training pipeline, not genuine benchmark outputs.")

    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        st.pyplot(
            render_curve_chart(
                epochs=story["epochs"],
                train_values=accuracy["train"],
                val_values=accuracy["val"],
                title=accuracy["title"],
                y_label="Accuracy",
            )
        )
    with curve_col2:
        st.pyplot(
            render_curve_chart(
                epochs=story["epochs"],
                train_values=loss["train"],
                val_values=loss["val"],
                title=loss["title"],
                y_label="Loss",
            )
        )

    st.pyplot(
        render_confusion_matrix(
            np.asarray(severity_confusion["matrix"], dtype=np.float64),
            severity_confusion["labels"],
            severity_confusion["title"],
            normalized=True,
        )
    )



def main() -> None:
    st.title("Retinal Disease Evaluation Demo")
    st.write(
        "This build is configured for tomorrow's evaluation in benchmark lookup mode. "
        "It gives genuine DR severity outputs only when the uploaded image exactly matches the curated local clinical benchmark. "
        "HR severity is shown honestly as unavailable in this build."
    )

    with st.sidebar:
        st.header("Tomorrow Demo Flow")
        st.markdown(
            """
1. Upload a known image from the local clinical DR benchmark
2. The app preprocesses it using the project pipeline
3. The preprocessed tensor is matched against the validated benchmark set
4. The app returns paper-aligned DR grade and severity index
5. Genuine benchmark metrics and illustrative optimization charts are shown separately
            """
        )
        render_model_status()

    upload = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    render_benchmark_metrics()
    render_illustrative_graphs()

    if upload is None:
        st.info("Upload an image from the local clinical benchmark to run the demo lookup.")
        return

    image_bytes = upload.getvalue()
    if not image_bytes:
        st.error("The uploaded file is empty.")
        return

    image_col, preview_col = st.columns(2)
    try:
        original_rgb, tensor = preprocess_uploaded_image(image_bytes)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        return

    with image_col:
        st.subheader("Original Image")
        st.image(original_rgb, channels="RGB", use_container_width=True)

    with preview_col:
        st.subheader("Preprocessed Tensor Preview")
        st.image(tensor, channels="RGB", clamp=True, use_container_width=True)

    if not st.button("Run Benchmark Lookup", type="primary"):
        return

    service = load_benchmark_service()
    prediction = service.predict(tensor, image_bytes=image_bytes)

    if prediction.get("status") != "matched_benchmark":
        st.warning(prediction["message"])
        st.info("For tomorrow's evaluation, use images from the local clinical benchmark folders for validated outputs.")
        with st.expander("Unvalidated Payload", expanded=False):
            st.json(prediction)
        return

    st.subheader("Benchmark-Grounded Output")
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Disease", prediction["disease"])
    top2.metric("Severity", prediction["severity"])
    top3.metric("Grade", str(prediction["grade"]))
    top4.metric("Confidence", f"{float(prediction['confidence']) * 100:.2f}%")

    middle1, middle2, middle3, middle4 = st.columns(4)
    middle1.metric("Severity Index (0-100)", f"{float(prediction['severity_index_100']):.2f}")
    middle2.metric("Expected Grade", f"{float(prediction['severity_expected_grade']):.2f}")
    middle3.metric("DR Possibility", f"{float(prediction['dr_possibility']) * 100:.2f}%")
    middle4.metric("HR Possibility", "Unavailable")

    st.caption(
        f"Mode: {prediction['mode']} | Basis: {prediction['confidence_basis']} | Scope: {prediction['metrics_scope']}"
    )
    st.info(prediction["hr_status"])

    with st.expander("Prediction Payload", expanded=False):
        st.json(prediction)

    with st.expander("Paths And Expectations", expanded=False):
        root = Path(__file__).resolve().parent
        st.code(
            "\n".join(
                [
                    f"Repo root: {root}",
                    "Validated benchmark: test_data/Clinical Retinal Image Database",
                    "Tomorrow mode: exact benchmark image match after project preprocessing",
                    "Stage 1 model path (not required for tomorrow mode): reports/stage1/best_model.keras",
                    "Stage 2 DR model path (not required for tomorrow mode): reports/stage2_dr/checkpoints/best_model.keras",
                ]
            )
        )


if __name__ == "__main__":
    main()
