# Retinal Disease Vision Transformer Pipeline

This project trains a two-stage retinal fundus pipeline for both diabetic retinopathy (DR) and hypertensive retinopathy (HR).

- Stage 1 predicts `normal`, `dr`, or `hr`.
- Stage 2 routes to the matching disease-specific severity grader.
- `normal` is the only source of final `Grade 0`.
- Stage 2 DR predicts grades `1-4`.
- Stage 2 HR predicts grades `1-4`.
- The final runtime payload reports `disease`, `severity`, `grade`, `confidence`, `stage1_confidence`, `stage2_confidence`, `stage1_probabilities`, and `stage2_probabilities`.

## Repo Layout

- `src/data/`: metadata building, preprocessing, and split generation
- `src/models/`: Stage 1 and Stage 2 ViT models
- `src/training/`: training entrypoints
- `src/eval/`: metrics and plots
- `src/inference/`: predictor and demo runtime
- `src/kaggle/`: Kaggle bundle export and artifact import tools
- `notebooks/kaggle_train.py`: Kaggle runner that stages dataset mounts and executes the pipeline
- `tests/`: non-TensorFlow contract and tooling tests
- `app.py`: Streamlit demo

## Setup

Local smoke checks and preprocessing require Python dependencies:

```bash
pip install -r requirements.txt
```

Full model training is expected to run on Kaggle GPU unless you have a local TensorFlow GPU environment and the raw datasets staged under `data/raw/`.

## Data Pipeline

1. Build unified metadata:

```bash
python src/data/build_master_metadata.py --config configs/data/data_config.yaml
```

2. Preprocess images into normalized `.npy` tensors:

```bash
python src/data/preprocess_images.py --config configs/data/data_config.yaml
```

3. Create leakage-safe `train / val / test` splits plus `external_test.csv`:

```bash
python src/data/make_splits.py --config configs/data/data_config.yaml
```

## Dataset Roles

- Training datasets use `dataset_role=train`.
- Curated benchmark datasets use `dataset_role=external_test`.
- External-test duplicates are excluded from train/val/test assignment.
- The local `Clinical Retinal Image Database` is parsed from diagnosis text and exported to `data/splits/external_test.csv` after quality filtering.

## Training

Train Stage 1:

```bash
python src/training/train_stage1.py --config configs/model_stage1.yaml
```

Train Stage 2 DR:

```bash
python src/training/train_stage2_dr.py --config configs/model_stage2.yaml
```

Train Stage 2 HR:

```bash
python src/training/train_stage2_hr.py --config configs/model_stage2.yaml
```

## Metrics And Reports

Stage 1 and Stage 2 runs write evaluation artifacts that include:

- accuracy and balanced accuracy
- macro and weighted precision / recall / F1
- sensitivity and specificity
- confusion matrices
- confidence summaries
- quadratic weighted kappa for ordered severity tasks

Expected model artifacts:

- `reports/stage1/best_model.keras`
- `reports/stage1/label_order.json`
- `reports/stage2_dr/checkpoints/best_model.keras`
- `reports/stage2_hr/checkpoints/best_model.keras`

## Kaggle Workflow

1. Export a clean Kaggle bundle from the local repo:

```bash
python src/kaggle/export_bundle.py --output-root artifacts/kaggle_bundle --include-test-data
```

This creates a parent folder containing:

- `wt-integrate/` with repo code, configs, notebooks, and tests
- `test_data/` as a sibling when the local clinical benchmark is available
- `bundle_manifest.json` with expected Kaggle dataset attachments and run commands

2. Upload the bundle to Kaggle and attach the training datasets.

3. Run the Kaggle pipeline script from the exported repo root:

```bash
python notebooks/kaggle_train.py           --combined-dr-root /kaggle/input/combined-dr-dataset-aptosidridmessidoreyepacs           --eyepacs-root /kaggle/input/eyepacs           --odir-root /kaggle/input/your-odir-dataset           --rvm-root /kaggle/input/your-rvm-dataset
```

4. Download the Kaggle output package and import the artifacts back into this repo:

```bash
python src/kaggle/import_artifacts.py --source path/to/kaggle_output
```

`RVM` remains the hard dependency for HR severity training. Without it, Stage 2 HR cannot complete.

## Streamlit Demo

Run:

```bash
streamlit run app.py
```

The app preprocesses the upload once and shows:

- disease
- severity
- grade
- overall confidence
- Stage 1 confidence
- Stage 2 confidence
- Stage 1 and Stage 2 probability maps

## Notes

- `image_path` may point to either a preprocessed `224x224x3` float32 normalized `.npy` tensor or a raw image file.
- `raw_image_path` preserves the original fundus image when preprocessing is used.
- Stage 2 severity is disease-specific, so DR and HR grades should not be compared as one shared ordinal target.
- If Kaggle credentials are configured locally, the added `kaggle` requirement lets you use the Kaggle CLI for dataset download, but the repo itself is designed to run the full training loop on Kaggle GPU.
