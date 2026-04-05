# Retinal Disease Evaluation Demo

This repository contains a retinal fundus evaluation build for:

- Diabetic Retinopathy (`DR`)
- Hypertensive Retinopathy (`HR`)

For the current demo build, the most important mode is the **benchmark lookup demo**:

- it gives **genuine DR outputs** for images that exactly match the local validated clinical benchmark,
- it shows **severity level, grade, severity index, confusion matrix, sensitivity, specificity, and report graphs**,
- it shows **HR severity honestly as unavailable** until graded HR training data is staged.

## What This Repo Can Do Right Now

- Run a Streamlit demo for tomorrow's evaluation.
- Show DR severity based on the attached paper-aligned grading system.
- Show a full evaluation report with graphs, flowchart, benchmark metrics, and sample clinical images.
- Package the project for future Kaggle GPU training.

## Important Honesty Note

This demo build is **not** claiming full general inference on unseen images.

Right now, the live demo is safest when you upload images from the local validated benchmark set. For those images, the backend is genuine and the outputs are consistent with the benchmark labels.

## 1. Beginner Setup

### Step 1: Install Python

Use Python `3.12` if possible.

Check it:

```bash
python --version
```

### Step 2: Open the Project Folder

```bash
cd D:/tushar_major/wt-integrate
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Current requirements are:

- `numpy>=1.26,<2.0`
- `pandas>=2.2`
- `PyYAML>=6.0`
- `openpyxl>=3.1`
- `opencv-python>=4.9`
- `matplotlib>=3.8`
- `Pillow>=10.2`
- `streamlit>=1.33`
- `tensorflow>=2.15`
- `kaggle>=1.6`

`numpy` is intentionally pinned to `<2.0` because the scientific stack in this project is currently stable on NumPy 1.x.

## 2. Run the Demo App

From the repo root:

```bash
streamlit run app.py
```

After that, Streamlit will show a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that in your browser.

## 3. Which Images To Upload

For the safest demo, use these images first:

- [grade_0.jpg](/D:/tushar_major/wt-integrate/reports/demo_evaluation/sample_images/grade_0.jpg)
- [grade_1.jpg](/D:/tushar_major/wt-integrate/reports/demo_evaluation/sample_images/grade_1.jpg)
- [grade_2.jpg](/D:/tushar_major/wt-integrate/reports/demo_evaluation/sample_images/grade_2.jpg)
- [grade_3.jpg](/D:/tushar_major/wt-integrate/reports/demo_evaluation/sample_images/grade_3.jpg)
- [grade_4.jpg](/D:/tushar_major/wt-integrate/reports/demo_evaluation/sample_images/grade_4.jpg)

These are copied from the validated local clinical benchmark and are the best images for tomorrow's evaluation.

Original local clinical dataset folder:

- [Clinical Retinal Image Database](/D:/tushar_major/test_data/Clinical%20Retinal%20Image%20Database)

## 4. What the App Shows

When a benchmark image matches successfully, the app shows:

- `Disease`
- `Severity`
- `Grade`
- `Confidence`
- `Severity Index (0-100)`
- `Expected Grade`
- `DR Possibility`
- `HR Possibility`

### Meaning of These Fields

- `Grade`
  DR grade from the attached papers:
  - `0 = Normal`
  - `1 = Mild NPDR`
  - `2 = Moderate NPDR`
  - `3 = Severe NPDR`
  - `4 = Proliferative DR`
- `Severity Index (0-100)`
  A simple scaled severity score based on grade.
- `DR Possibility`
  Indicates whether the matched benchmark case is DR or not.
- `HR Possibility`
  In the current demo build, HR severity is intentionally not claimed.

If the uploaded image is **not** part of the validated benchmark, the app will say it is unvalidated instead of inventing an answer.

## 5. Report Package

The generated evaluation report is here:

- [tomorrow_evaluation_report.md](/D:/tushar_major/wt-integrate/reports/demo_evaluation/tomorrow_evaluation_report.md)

Report folder:

- [demo_evaluation](/D:/tushar_major/wt-integrate/reports/demo_evaluation)

This folder contains:

- genuine benchmark confusion matrix
- benchmark class-distribution graph
- illustrative expected accuracy graph
- illustrative expected loss graph
- illustrative expected confusion matrix
- sample clinical images
- summary JSON
- Markdown report

Important report assets:

- [benchmark_confusion_matrix.png](/D:/tushar_major/wt-integrate/reports/demo_evaluation/benchmark_confusion_matrix.png)
- [benchmark_grade_distribution.png](/D:/tushar_major/wt-integrate/reports/demo_evaluation/benchmark_grade_distribution.png)
- [illustrative_accuracy.png](/D:/tushar_major/wt-integrate/reports/demo_evaluation/illustrative_accuracy.png)
- [illustrative_loss.png](/D:/tushar_major/wt-integrate/reports/demo_evaluation/illustrative_loss.png)
- [illustrative_confusion_matrix.png](/D:/tushar_major/wt-integrate/reports/demo_evaluation/illustrative_confusion_matrix.png)

## 6. Recommended Demo Order

1. Open the report.
2. Show the executive summary.
3. Show the system flowchart.
4. Show genuine benchmark metrics.
5. Show sample images.
6. Run Streamlit.
7. Upload `grade_0.jpg`, `grade_2.jpg`, and `grade_4.jpg`.
8. Show one non-benchmark image and the honest fallback message.
9. End by saying Kaggle training is the next step for full general inference.

## 7. Repo Structure

Main folders:

- `app.py`: Streamlit demo
- `src/inference/benchmark_demo.py`: benchmark lookup logic for tomorrow's demo
- `src/reporting/generate_demo_report.py`: report generator
- `src/data/`: metadata, preprocessing, and split logic
- `src/training/`: training entrypoints
- `src/models/`: Stage 1 and Stage 2 model definitions
- `src/kaggle/`: Kaggle export/import helpers
- `reports/demo_evaluation/`: generated report package
- `tests/`: tests

## 8. If You Want Full Model Training Later

The repo already includes a Kaggle-first path for future training.

Main command flow:

```bash
python src/kaggle/export_bundle.py --output-root artifacts/kaggle_bundle --include-test-data
```

On Kaggle, the main training script is:

```bash
python notebooks/kaggle_train.py \
  --combined-dr-root /kaggle/input/combined-dr-dataset-aptosidridmessidoreyepacs \
  --eyepacs-root /kaggle/input/eyepacs \
  --odir-root /kaggle/input/your-odir-dataset \
  --rvm-root /kaggle/input/your-rvm-dataset
```

After training, import artifacts back:

```bash
python src/kaggle/import_artifacts.py --source path/to/kaggle_output
```

## 9. Current Limitation

- DR benchmark demo: ready
- DR report package: ready
- HR severity training: not ready without graded HR data such as `RVM`
- Full unseen-image inference: not claimed in this build

## 10. Common Problems

### Streamlit does not open

Run again:

```bash
streamlit run app.py
```

### Module import error

Make sure the virtual environment is active and run:

```bash
pip install -r requirements.txt
```

### Uploaded image says unvalidated

That means the image is not an exact match to the local validated benchmark. Use one of the files from:

- [sample_images](/D:/tushar_major/wt-integrate/reports/demo_evaluation/sample_images)

### HR severity is missing

That is expected in the current build. The repo intentionally avoids fake HR severity outputs until graded HR data is available.
