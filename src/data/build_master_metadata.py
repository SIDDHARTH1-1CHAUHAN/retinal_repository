from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.contracts import (
    MASTER_METADATA_PATH,
    ensure_metadata_contract,
    load_yaml_config,
    make_safe_image_id,
    read_table,
    resolve_repo_path,
    save_metadata,
)

IMAGE_EXTENSIONS = [".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"]
LEFT_RIGHT_SUFFIX_RE = re.compile(r"(.+?)(?:[_-](left|right|l|r))?$", re.IGNORECASE)
GRADE_RE = re.compile(r"(?:stage|grade)\s*([0-4])", re.IGNORECASE)
PROLIFERATIVE_RE = re.compile(r"(?:onset\s+)?proliferative\s+diabetic\s+retinopathy", re.IGNORECASE)
LESION_KEYWORDS = ("exudate", "hemor", "haemor", "cotton", "pdr")
AUXILIARY_IMAGE_KEYWORDS = ("_gt", "disk", "od_", "_od", "bv", "vessel", "exudate", "hemor", "haemor", "cotton", "pdr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical retinal metadata contract.")
    parser.add_argument("--config", default="configs/data/data_config.yaml", help="Path to data registry YAML.")
    parser.add_argument("--output", default=str(MASTER_METADATA_PATH), help="Output metadata CSV path.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing dataset paths.")
    return parser.parse_args()


def find_first_column(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    alias_map = {column.lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in alias_map:
            return alias_map[alias.lower()]
    if required:
        raise KeyError(f"Could not resolve any of these columns: {list(aliases)}")
    return None


def resolve_image_path(image_dir: Path, source_image_id: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            candidate = image_dir / explicit_path
            if candidate.exists():
                return candidate

    source_path = Path(source_image_id)
    if source_path.suffix:
        candidates = [image_dir / source_path.name, image_dir / source_image_id]
    else:
        candidates = [image_dir / f"{source_image_id}{extension}" for extension in IMAGE_EXTENSIONS]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if image_dir.exists():
        stem = source_path.stem if source_path.suffix else source_image_id
        matches = []
        for extension in IMAGE_EXTENSIONS:
            matches.extend(image_dir.glob(f"{stem}{extension}"))
            matches.extend(image_dir.glob(f"{stem}{extension.upper()}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not resolve image for id '{source_image_id}' under {image_dir}")


def derive_patient_id(source_image_id: str, explicit_patient_id: Any = None) -> str:
    if explicit_patient_id is not None and not pd.isna(explicit_patient_id) and str(explicit_patient_id).strip():
        return str(explicit_patient_id).strip()
    stem = Path(str(source_image_id)).stem
    match = LEFT_RIGHT_SUFFIX_RE.match(stem)
    if not match:
        return stem
    return match.group(1) or stem


def merge_quality_flags(*flags: str) -> str:
    ordered = []
    for flag in flags:
        normalized = str(flag or "").strip().lower()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return "|".join(ordered)


def finalize_row(row: dict[str, Any], dataset_cfg: dict[str, Any], default_label_source: str, quality_flag: str = "") -> dict[str, Any]:
    row["split"] = ""
    row["is_manual_label"] = bool(dataset_cfg.get("is_manual_label", True))
    row["dataset_role"] = str(dataset_cfg.get("dataset_role", "train")).strip().lower()
    row["label_source"] = str(row.get("label_source") or dataset_cfg.get("label_source", default_label_source)).strip()
    row["quality_flag"] = merge_quality_flags(dataset_cfg.get("quality_flag", ""), row.get("quality_flag", ""), quality_flag)
    row.setdefault("sha256", "")
    row.setdefault("dhash", "")
    return row


def parse_dr_grading_dataset(dataset_name: str, dataset_cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    labels_path = resolve_repo_path(project_root, dataset_cfg["labels_csv"])
    image_dir = resolve_repo_path(project_root, dataset_cfg["image_dir"])
    frame = read_table(labels_path)

    image_column = dataset_cfg.get("image_column") or find_first_column(frame, ["image", "image_id", "id_code", "filename"])
    grade_column = dataset_cfg.get("grade_column") or find_first_column(frame, ["level", "diagnosis", "dr_grade", "retinopathy_grade", "grade"])
    patient_column = dataset_cfg.get("patient_id_column") or find_first_column(frame, ["patient_or_case_id", "patient_id", "patient", "case_id"], required=False)
    path_column = dataset_cfg.get("path_column") or find_first_column(frame, ["image_path", "filepath", "path"], required=False)

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        source_image_id = str(getattr(row, image_column)).strip()
        dr_grade = int(float(getattr(row, grade_column)))
        image_path = resolve_image_path(
            image_dir=image_dir,
            source_image_id=source_image_id,
            explicit_path=getattr(row, path_column) if path_column else None,
        )
        patient_id = derive_patient_id(source_image_id, getattr(row, patient_column) if patient_column else None)
        rows.append(
            finalize_row(
                {
                    "image_id": make_safe_image_id(dataset_name, source_image_id),
                    "source_image_id": source_image_id,
                    "source_dataset": dataset_name,
                    "patient_or_case_id": patient_id,
                    "image_path": str(image_path.resolve()),
                    "raw_image_path": str(image_path.resolve()),
                    "disease_label": "normal" if dr_grade == 0 else "dr",
                    "dr_grade": dr_grade,
                    "hr_grade": pd.NA,
                },
                dataset_cfg,
                default_label_source="grading_csv",
            )
        )
    return rows


def infer_odir_label(row: pd.Series, side_prefix: str) -> str | None:
    keyword_column = None
    for candidate in [f"{side_prefix}_diagnostic_keywords", f"{side_prefix}_diagnosis", f"{side_prefix}_keywords"]:
        if candidate in row.index:
            keyword_column = candidate
            break

    text = ""
    if keyword_column:
        raw_text = row.get(keyword_column)
        text = "" if pd.isna(raw_text) else str(raw_text).strip().lower()

    def positive(*aliases: str) -> bool:
        for alias in aliases:
            if alias in row.index and not pd.isna(row[alias]) and int(float(row[alias])) == 1:
                return True
        return False

    has_hr = positive("hypertension", "hypertensive_retinopathy") or "hypertensive" in text
    has_dr = positive("diabetic_retinopathy", "dr") or "diabetic retinopathy" in text
    has_normal = positive("normal_fundus", "normal") or text == "normal fundus"

    positive_targets = [label for label, present in [("hr", has_hr), ("dr", has_dr), ("normal", has_normal)] if present]
    if len(positive_targets) != 1:
        return None
    return positive_targets[0]


def parse_odir_dataset(dataset_name: str, dataset_cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    labels_path = resolve_repo_path(project_root, dataset_cfg["labels_csv"])
    image_dir = resolve_repo_path(project_root, dataset_cfg["image_dir"])
    frame = read_table(labels_path)
    frame.columns = [column.strip().lower().replace("-", "_").replace(" ", "_") for column in frame.columns]

    patient_column = find_first_column(frame, ["patient_or_case_id", "patient_id", "id", "case_id"])
    left_image_column = find_first_column(frame, ["left_fundus", "left_image", "left_filename"])
    right_image_column = find_first_column(frame, ["right_fundus", "right_image", "right_filename"])

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        normalized_row = pd.Series(record)
        patient_id = str(normalized_row[patient_column]).strip()
        for side_prefix, image_column in [("left", left_image_column), ("right", right_image_column)]:
            source_image_id = str(normalized_row[image_column]).strip()
            if not source_image_id:
                continue
            disease_label = infer_odir_label(normalized_row, side_prefix=side_prefix)
            if disease_label is None:
                continue
            image_path = resolve_image_path(image_dir=image_dir, source_image_id=source_image_id)
            rows.append(
                finalize_row(
                    {
                        "image_id": make_safe_image_id(dataset_name, source_image_id),
                        "source_image_id": source_image_id,
                        "source_dataset": dataset_name,
                        "patient_or_case_id": patient_id,
                        "image_path": str(image_path.resolve()),
                        "raw_image_path": str(image_path.resolve()),
                        "disease_label": disease_label,
                        "dr_grade": 0 if disease_label == "normal" else pd.NA,
                        "hr_grade": pd.NA,
                    },
                    dataset_cfg,
                    default_label_source="odir_sheet",
                )
            )
    return rows


def parse_hr_grading_dataset(dataset_name: str, dataset_cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    labels_path = resolve_repo_path(project_root, dataset_cfg["labels_csv"])
    image_dir = resolve_repo_path(project_root, dataset_cfg["image_dir"])
    frame = read_table(labels_path)

    image_column = dataset_cfg.get("image_column") or find_first_column(frame, ["image", "image_id", "id_code", "filename", "file_name"])
    grade_column = dataset_cfg.get("grade_column") or find_first_column(frame, ["hr_grade", "severity", "grade", "label"])
    patient_column = dataset_cfg.get("patient_id_column") or find_first_column(frame, ["patient_or_case_id", "patient_id", "patient", "case_id"], required=False)
    path_column = dataset_cfg.get("path_column") or find_first_column(frame, ["image_path", "filepath", "path"], required=False)

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        source_image_id = str(getattr(row, image_column)).strip()
        hr_grade = int(float(getattr(row, grade_column)))
        if hr_grade not in {1, 2, 3, 4}:
            continue
        image_path = resolve_image_path(
            image_dir=image_dir,
            source_image_id=source_image_id,
            explicit_path=getattr(row, path_column) if path_column else None,
        )
        patient_id = derive_patient_id(source_image_id, getattr(row, patient_column) if patient_column else None)
        rows.append(
            finalize_row(
                {
                    "image_id": make_safe_image_id(dataset_name, source_image_id),
                    "source_image_id": source_image_id,
                    "source_dataset": dataset_name,
                    "patient_or_case_id": patient_id,
                    "image_path": str(image_path.resolve()),
                    "raw_image_path": str(image_path.resolve()),
                    "disease_label": "hr",
                    "dr_grade": pd.NA,
                    "hr_grade": hr_grade,
                },
                dataset_cfg,
                default_label_source="grading_csv",
            )
        )
    return rows


def parse_support_binary_dataset(dataset_name: str, dataset_cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    labels_path = resolve_repo_path(project_root, dataset_cfg["labels_csv"])
    image_dir = resolve_repo_path(project_root, dataset_cfg["image_dir"])
    frame = read_table(labels_path)

    image_column = dataset_cfg.get("image_column") or find_first_column(frame, ["image", "image_id", "filename", "file_name"])
    label_column = dataset_cfg.get("label_column") or find_first_column(frame, ["label", "diagnosis", "class", "disease_label"])
    patient_column = dataset_cfg.get("patient_id_column") or find_first_column(frame, ["patient_or_case_id", "patient_id", "patient", "case_id"], required=False)
    positive_label = dataset_cfg.get("positive_label", "dr")

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        source_image_id = str(getattr(row, image_column)).strip()
        raw_label = str(getattr(row, label_column)).strip().lower()
        disease_label = positive_label if raw_label in {"1", positive_label, "positive", "abnormal"} else "normal"
        image_path = resolve_image_path(image_dir=image_dir, source_image_id=source_image_id)
        patient_id = derive_patient_id(source_image_id, getattr(row, patient_column) if patient_column else None)
        rows.append(
            finalize_row(
                {
                    "image_id": make_safe_image_id(dataset_name, source_image_id),
                    "source_image_id": source_image_id,
                    "source_dataset": dataset_name,
                    "patient_or_case_id": patient_id,
                    "image_path": str(image_path.resolve()),
                    "raw_image_path": str(image_path.resolve()),
                    "disease_label": disease_label,
                    "dr_grade": 0 if disease_label == "normal" else pd.NA,
                    "hr_grade": pd.NA,
                },
                dataset_cfg,
                default_label_source="binary_label_file",
            )
        )
    return rows


def parse_manifest_dataset(dataset_name: str, dataset_cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_repo_path(project_root, dataset_cfg["manifest_path"])
    base_dir = resolve_repo_path(project_root, dataset_cfg.get("base_dir", manifest_path.parent))
    frame = read_table(manifest_path)
    required_columns = {"source_image_id", "image_path", "disease_label", "dr_grade", "hr_grade"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest dataset '{dataset_name}' is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        source_image_id = str(record.get("source_image_id") or record.get("image_id") or "").strip()
        if not source_image_id:
            continue
        image_path = Path(str(record["image_path"]))
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        raw_image_path = Path(str(record.get("raw_image_path") or image_path))
        if not raw_image_path.is_absolute():
            raw_image_path = base_dir / raw_image_path
        rows.append(
            finalize_row(
                {
                    "image_id": str(record.get("image_id") or make_safe_image_id(dataset_name, source_image_id)).strip(),
                    "source_image_id": source_image_id,
                    "source_dataset": str(record.get("source_dataset") or dataset_name).strip().lower(),
                    "patient_or_case_id": str(record.get("patient_or_case_id") or derive_patient_id(source_image_id)).strip(),
                    "image_path": str(image_path.resolve()),
                    "raw_image_path": str(raw_image_path.resolve()),
                    "disease_label": str(record["disease_label"]).strip().lower(),
                    "dr_grade": record.get("dr_grade", pd.NA),
                    "hr_grade": record.get("hr_grade", pd.NA),
                    "dataset_role": str(record.get("dataset_role") or dataset_cfg.get("dataset_role", "train")).strip().lower(),
                    "label_source": str(record.get("label_source") or dataset_cfg.get("label_source", "manifest")).strip(),
                    "quality_flag": merge_quality_flags(dataset_cfg.get("quality_flag", ""), str(record.get("quality_flag") or "")),
                    "sha256": str(record.get("sha256") or "").strip().lower(),
                    "dhash": str(record.get("dhash") or "").strip().lower(),
                },
                dataset_cfg,
                default_label_source="manifest",
                quality_flag=str(record.get("quality_flag") or ""),
            )
        )
    return rows


def parse_clinical_diagnosis_text(text: str) -> tuple[int | None, str]:
    normalized = str(text or "").strip()
    match = GRADE_RE.search(normalized)
    if match:
        return int(match.group(1)), normalized
    if PROLIFERATIVE_RE.search(normalized):
        return 4, normalized
    lowered = normalized.lower()
    if "no dr lesions present" in lowered or "normal fundus" in lowered:
        return 0, normalized
    return None, normalized


def choose_clinical_fundus_image(case_dir: Path) -> Path | None:
    candidates = [path for path in case_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    primary: list[Path] = []
    for candidate in candidates:
        name = candidate.name.lower()
        if any(keyword in name for keyword in AUXILIARY_IMAGE_KEYWORDS):
            continue
        primary.append(candidate)
    if primary:
        primary.sort(key=lambda path: (len(path.name), path.name.lower()))
        return primary[0]
    return None


def clinical_has_lesion_masks(case_dir: Path) -> bool:
    for path in case_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(keyword in name for keyword in LESION_KEYWORDS):
            return True
    return False


def parse_clinical_dr_dataset(dataset_name: str, dataset_cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    dataset_dir = resolve_repo_path(project_root, dataset_cfg["dataset_dir"])
    rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        diagnosis_files = sorted(case_dir.glob("*Diagnosis.txt"))
        diagnosis_text = diagnosis_files[0].read_text(encoding="utf-8", errors="ignore") if diagnosis_files else ""
        dr_grade, _ = parse_clinical_diagnosis_text(diagnosis_text)
        fundus_image = choose_clinical_fundus_image(case_dir)

        quality_flag = ""
        if not diagnosis_files:
            quality_flag = merge_quality_flags(quality_flag, "missing_diagnosis")
        if dr_grade is None:
            quality_flag = merge_quality_flags(quality_flag, "unknown_grade")
        if fundus_image is None:
            quality_flag = merge_quality_flags(quality_flag, "missing_fundus_image")
        if dr_grade == 0 and clinical_has_lesion_masks(case_dir):
            quality_flag = merge_quality_flags(quality_flag, "ambiguous_stage0_with_lesions")
        if diagnosis_files:
            expected_token = re.sub(r"[^0-9]", "", case_dir.name)
            diagnosis_token = re.sub(r"[^0-9]", "", diagnosis_files[0].stem)
            if expected_token and diagnosis_token and expected_token != diagnosis_token:
                quality_flag = merge_quality_flags(quality_flag, "diagnosis_filename_mismatch")

        if dataset_cfg.get("skip_unknown_grades", True) and dr_grade is None:
            continue
        if fundus_image is None:
            continue

        rows.append(
            finalize_row(
                {
                    "image_id": make_safe_image_id(dataset_name, case_dir.name),
                    "source_image_id": fundus_image.name,
                    "source_dataset": dataset_name,
                    "patient_or_case_id": case_dir.name.strip(),
                    "image_path": str(fundus_image.resolve()),
                    "raw_image_path": str(fundus_image.resolve()),
                    "disease_label": "normal" if int(dr_grade or 0) == 0 else "dr",
                    "dr_grade": int(dr_grade or 0),
                    "hr_grade": pd.NA,
                    "label_source": "diagnosis_text",
                    "quality_flag": quality_flag,
                },
                dataset_cfg,
                default_label_source="diagnosis_text",
                quality_flag=quality_flag,
            )
        )
    return rows


PARSERS = {
    "dr_grading": parse_dr_grading_dataset,
    "eyepacs": parse_dr_grading_dataset,
    "aptos": parse_dr_grading_dataset,
    "messidor": parse_dr_grading_dataset,
    "idrid": parse_dr_grading_dataset,
    "odir": parse_odir_dataset,
    "rvm_hr": parse_hr_grading_dataset,
    "support_binary": parse_support_binary_dataset,
    "manifest": parse_manifest_dataset,
    "clinical_dr_text": parse_clinical_dr_dataset,
}


def dataset_inputs_exist(parser_name: str, dataset_cfg: dict[str, Any], project_root: Path) -> tuple[bool, str]:
    if parser_name == "manifest":
        manifest_path = resolve_repo_path(project_root, dataset_cfg["manifest_path"])
        return manifest_path.exists(), f"manifest={manifest_path}"
    if parser_name == "clinical_dr_text":
        dataset_dir = resolve_repo_path(project_root, dataset_cfg["dataset_dir"])
        return dataset_dir.exists(), f"dataset_dir={dataset_dir}"

    labels_path = resolve_repo_path(project_root, dataset_cfg["labels_csv"])
    image_dir = resolve_repo_path(project_root, dataset_cfg["image_dir"])
    return labels_path.exists() and image_dir.exists(), f"labels={labels_path}, images={image_dir}"


def build_master_dataframe(config: dict[str, Any], project_root: Path, strict: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets = config.get("datasets", {})
    if not datasets:
        raise ValueError("No datasets configured in data_config.yaml")

    for dataset_name, dataset_cfg in datasets.items():
        if not dataset_cfg.get("enabled", False):
            continue
        parser_name = dataset_cfg["parser"]
        parser = PARSERS[parser_name]
        dataset_required = bool(dataset_cfg.get("required", False))
        inputs_exist, description = dataset_inputs_exist(parser_name, dataset_cfg, project_root)
        if not inputs_exist:
            message = f"Missing required inputs for dataset '{dataset_name}' ({description})"
            if strict or dataset_required:
                raise FileNotFoundError(message)
            print(f"Skipping optional dataset: {message}")
            continue
        rows.extend(parser(dataset_name=dataset_name, dataset_cfg=dataset_cfg, project_root=project_root))

    if not rows:
        raise RuntimeError("No metadata rows were produced. Check dataset paths and parser settings.")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    return ensure_metadata_contract(frame)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(resolve_repo_path(project_root, args.config))
    metadata = build_master_dataframe(config=config, project_root=project_root, strict=args.strict)
    output_path = save_metadata(metadata, resolve_repo_path(project_root, args.output))
    print(f"Wrote {len(metadata)} rows to {output_path}")


if __name__ == "__main__":
    main()
