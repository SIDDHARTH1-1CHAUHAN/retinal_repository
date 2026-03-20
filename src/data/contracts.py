from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

MANDATORY_METADATA_COLUMNS = [
    "image_id",
    "source_dataset",
    "patient_or_case_id",
    "image_path",
    "disease_label",
    "dr_grade",
    "hr_grade",
    "split",
    "is_manual_label",
]

OPTIONAL_METADATA_COLUMNS = [
    "source_image_id",
    "raw_image_path",
    "duplicate_group_id",
]

MASTER_METADATA_PATH = Path("data/metadata/master.csv")
SPLIT_OUTPUT_PATHS = {
    "train": Path("data/splits/train.csv"),
    "val": Path("data/splits/val.csv"),
    "test": Path("data/splits/test.csv"),
}

STAGE1_LABEL_TO_INDEX = {"normal": 0, "dr": 1, "hr": 2}
DR_STAGE2_CLASSES = [0, 1, 2, 3, 4]
HR_STAGE2_CLASSES = [1, 2, 3, 4]
FINAL_PREDICTION_FIELDS = ["disease", "severity", "confidence"]

ALLOWED_DISEASE_LABELS = set(STAGE1_LABEL_TO_INDEX)
ALLOWED_SPLITS = {"", "train", "val", "test"}


def is_stage1_eligible_row(row: pd.Series) -> bool:
    disease_label = str(row.get("disease_label", "")).strip().lower()
    dr_grade = normalize_optional_int(row.get("dr_grade"))
    hr_grade = normalize_optional_int(row.get("hr_grade"))
    return disease_label in ALLOWED_DISEASE_LABELS and not (
        pd.notna(dr_grade) and pd.notna(hr_grade)
    )


def is_stage2_dr_eligible_row(row: pd.Series) -> bool:
    disease_label = str(row.get("disease_label", "")).strip().lower()
    dr_grade = normalize_optional_int(row.get("dr_grade"))
    hr_grade = normalize_optional_int(row.get("hr_grade"))
    return disease_label in {"normal", "dr"} and pd.notna(dr_grade) and pd.isna(hr_grade)


def is_stage2_hr_eligible_row(row: pd.Series) -> bool:
    disease_label = str(row.get("disease_label", "")).strip().lower()
    dr_grade = normalize_optional_int(row.get("dr_grade"))
    hr_grade = normalize_optional_int(row.get("hr_grade"))
    return disease_label == "hr" and pd.isna(dr_grade) and pd.notna(hr_grade)


def stage1_eligible_mask(df: pd.DataFrame) -> pd.Series:
    return df.apply(is_stage1_eligible_row, axis=1)


def stage2_dr_eligible_mask(df: pd.DataFrame) -> pd.Series:
    return df.apply(is_stage2_dr_eligible_row, axis=1)


def stage2_hr_eligible_mask(df: pd.DataFrame) -> pd.Series:
    return df.apply(is_stage2_hr_eligible_row, axis=1)


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def make_safe_image_id(source_dataset: str, source_image_id: str) -> str:
    joined = f"{source_dataset}__{source_image_id}"
    safe_chars = []
    for char in joined:
        if char.isalnum() or char in {"_", "-", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def normalize_optional_int(value: Any) -> int | pd.NA:
    if value is None:
        return pd.NA
    if isinstance(value, str) and not value.strip():
        return pd.NA
    if pd.isna(value):
        return pd.NA
    return int(float(value))


def ensure_metadata_contract(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MANDATORY_METADATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing mandatory metadata columns: {missing}")

    normalized = df.copy()
    normalized["image_id"] = normalized["image_id"].astype(str).str.strip()
    normalized["source_dataset"] = normalized["source_dataset"].astype(str).str.strip().str.lower()
    normalized["patient_or_case_id"] = normalized["patient_or_case_id"].fillna("").astype(str).str.strip()
    normalized["image_path"] = normalized["image_path"].astype(str).str.strip()
    normalized["disease_label"] = normalized["disease_label"].astype(str).str.strip().str.lower()
    normalized["split"] = normalized["split"].fillna("").astype(str).str.strip().str.lower()
    normalized["is_manual_label"] = normalized["is_manual_label"].astype(bool)
    normalized["dr_grade"] = normalized["dr_grade"].apply(normalize_optional_int).astype("Int64")
    normalized["hr_grade"] = normalized["hr_grade"].apply(normalize_optional_int).astype("Int64")

    invalid_labels = sorted(set(normalized["disease_label"]) - ALLOWED_DISEASE_LABELS)
    if invalid_labels:
        raise ValueError(f"Unsupported disease labels: {invalid_labels}")

    invalid_splits = sorted(set(normalized["split"]) - ALLOWED_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported split values: {invalid_splits}")

    if normalized["image_id"].duplicated().any():
        duplicates = normalized.loc[normalized["image_id"].duplicated(), "image_id"].tolist()[:10]
        raise ValueError(f"Duplicate image_id values detected: {duplicates}")

    if not normalized["dr_grade"].dropna().isin(DR_STAGE2_CLASSES).all():
        raise ValueError(f"dr_grade values must be within {DR_STAGE2_CLASSES}")
    if not normalized["hr_grade"].dropna().isin(HR_STAGE2_CLASSES).all():
        raise ValueError(f"hr_grade values must be within {HR_STAGE2_CLASSES}")

    normal_rows = normalized["disease_label"] == "normal"
    dr_rows = normalized["disease_label"] == "dr"
    hr_rows = normalized["disease_label"] == "hr"

    if (normalized["dr_grade"].notna() & normalized["hr_grade"].notna()).any():
        raise ValueError("A row cannot have both dr_grade and hr_grade populated")
    if normalized.loc[normal_rows, "hr_grade"].notna().any():
        raise ValueError("normal samples cannot have hr_grade")
    if normalized.loc[hr_rows, "dr_grade"].notna().any():
        raise ValueError("hr samples cannot have dr_grade")
    if normalized.loc[dr_rows, "hr_grade"].notna().any():
        raise ValueError("dr samples cannot have hr_grade")
    if not normalized.loc[normal_rows, "dr_grade"].dropna().isin([0]).all():
        raise ValueError("normal samples may only use dr_grade=0 or null")

    ordered_columns = MANDATORY_METADATA_COLUMNS + [
        column for column in OPTIONAL_METADATA_COLUMNS if column in normalized.columns
    ]
    trailing_columns = [column for column in normalized.columns if column not in ordered_columns]
    return normalized.loc[:, ordered_columns + trailing_columns].sort_values(
        ["source_dataset", "patient_or_case_id", "image_id"]
    ).reset_index(drop=True)


def save_metadata(df: pd.DataFrame, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_metadata_contract(df).to_csv(output_path, index=False)
    return output_path


def resolve_repo_path(project_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def load_image_tensor(path: str | Path) -> np.ndarray:
    array = np.load(path)
    if array.shape != (224, 224, 3):
        raise ValueError(f"Expected tensor shape (224, 224, 3), got {array.shape} for {path}")
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return array
