from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.inference.demo_runtime import (
    load_predictor,
    model_availability,
    preprocess_uploaded_image,
)


st.set_page_config(page_title="Retinal ViT Demo", layout="wide")


def render_model_status() -> None:
    st.subheader("Model Status")
    status = model_availability()
    for name, path in status.items():
        if str(path) and path.exists():
            st.success(f"{name}: found at {path}")
        else:
            st.warning(f"{name}: missing at {path}")


def main() -> None:
    st.title("Retinal Disease Vision Transformer Demo")
    st.write(
        "Upload a fundus image to run the integrated two-stage pipeline. "
        "The app preprocesses the image, predicts the disease with Stage 1, "
        "then routes to the matching severity grader."
    )

    with st.sidebar:
        st.header("Project Flow")
        st.markdown(
            """
1. Upload fundus image
2. Crop + CLAHE + resize + normalize
3. Stage 1 ViT predicts `Normal`, `DR`, or `HR`
4. Stage 2 ViT predicts disease-specific severity
5. App displays disease, severity, and confidence
            """
        )
        render_model_status()

    upload = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    if upload is None:
        st.info("Upload an image to start the demo.")
        return

    image_bytes = upload.getvalue()
    if not image_bytes:
        st.error("The uploaded file is empty.")
        return

    col1, col2 = st.columns(2)

    try:
        original_rgb, tensor = preprocess_uploaded_image(image_bytes)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        return

    with col1:
        st.subheader("Original Image")
        st.image(original_rgb, channels="RGB", use_container_width=True)

    with col2:
        st.subheader("Preprocessed Tensor Preview")
        st.image(tensor, channels="RGB", clamp=True, use_container_width=True)

    if not st.button("Run Prediction", type="primary"):
        return

    try:
        predictor = load_predictor()
        prediction = predictor.predict(tensor)
        details = predictor.predict_with_details(tensor)
    except Exception as exc:
        st.error(
            "Prediction failed. Make sure TensorFlow is installed and the Stage 1 / Stage 2 model files exist.\n\n"
            f"Details: {exc}"
        )
        return

    st.subheader("Prediction")
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Disease", prediction["disease"])
    result_col2.metric("Severity", prediction["severity"])
    result_col3.metric("Confidence", f"{float(prediction['confidence']) * 100:.2f}%")

    stage1_disease = details.get("stage1_disease", "unknown")
    grade = details.get("grade")
    st.caption(f"Stage 1 routing label: {stage1_disease} | Numeric grade: {grade}")

    with st.expander("Paths and Model Expectations"):
        root = Path(__file__).resolve().parent
        st.code(
            "\n".join(
                [
                    f"Repo root: {root}",
                    "Stage 1 model: reports/stage1/best_model.keras",
                    "Stage 2 DR model: reports/stage2_dr/checkpoints/best_model.keras",
                    "Stage 2 HR model: reports/stage2_hr/checkpoints/best_model.keras",
                ]
            )
        )


if __name__ == "__main__":
    main()
