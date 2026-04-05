from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.contracts import (
    EXTERNAL_TEST_OUTPUT_PATH,
    MASTER_METADATA_PATH,
    SPLIT_OUTPUT_PATHS,
    load_image_tensor,
    load_yaml_config,
    resolve_repo_path,
    save_metadata,
)


def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for duplicate hashing. Install opencv-python before running this script."
        ) from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-safe train/val/test CSV files.")
    parser.add_argument("--config", default="configs/data/data_config.yaml", help="Path to the data config.")
    parser.add_argument("--metadata", default=str(MASTER_METADATA_PATH), help="Input master metadata CSV.")
    return parser.parse_args()


def sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_from_array(array: np.ndarray) -> str:
    cv2 = require_cv2()
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if bit else "0" for bit in diff.flatten())
    return f"{int(bits, 2):016x}"


def append_quality_flag(current: str, flag: str) -> str:
    values = [value.strip().lower() for value in str(current or "").split("|") if value.strip()]
    normalized_flag = str(flag or "").strip().lower()
    if normalized_flag and normalized_flag not in values:
        values.append(normalized_flag)
    return "|".join(values)


def load_hashable_image(row: pd.Series) -> np.ndarray | None:
    cv2 = require_cv2()
    image_path = Path(str(row.get("image_path", "")))
    raw_image_path = Path(str(row.get("raw_image_path", "")))
    if image_path.suffix.lower() == ".npy" and image_path.exists():
        return load_image_tensor(image_path)
    if raw_image_path.exists():
        raw_bgr = cv2.imread(str(raw_image_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            return None
        return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    if image_path.exists():
        raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            return None
        return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    return None


def enrich_hash_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "sha256" not in enriched.columns:
        enriched["sha256"] = ""
    if "dhash" not in enriched.columns:
        enriched["dhash"] = ""

    for index, row in enriched.iterrows():
        file_path = Path(str(row.get("raw_image_path") or row.get("image_path") or ""))
        if file_path.exists() and not str(enriched.at[index, "sha256"]).strip():
            enriched.at[index, "sha256"] = sha256_for_path(file_path)
        if not str(enriched.at[index, "dhash"]).strip():
            image = load_hashable_image(row)
            if image is not None:
                enriched.at[index, "dhash"] = dhash_from_array(image)
    return enriched


def enrich_duplicate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_hash_columns(frame)
    duplicate_map: dict[str, list[int]] = defaultdict(list)
    for index, row in enriched.iterrows():
        hash_key = str(row.get("dhash", "")).strip().lower()
        if hash_key:
            duplicate_map[hash_key].append(index)

    enriched["duplicate_group_id"] = pd.NA
    for hash_key, indices in duplicate_map.items():
        if len(indices) >= 2:
            enriched.loc[indices, "duplicate_group_id"] = f"dup_{hash_key}"
    return enriched


def choose_group_anchor(row: pd.Series) -> str:
    duplicate_group_id = row.get("duplicate_group_id", pd.NA)
    if pd.notna(duplicate_group_id):
        return str(duplicate_group_id)
    patient_id = str(row.get("patient_or_case_id", "")).strip()
    if patient_id:
        return patient_id
    return str(row["image_id"])


def severity_bucket(row: pd.Series) -> str:
    disease_label = str(row.get("disease_label", "")).strip().lower()
    dr_grade = pd.to_numeric(row.get("dr_grade"), errors="coerce")
    hr_grade = pd.to_numeric(row.get("hr_grade"), errors="coerce")
    if disease_label == "dr" and pd.notna(dr_grade):
        return f"dr_grade_{int(dr_grade)}"
    if disease_label == "hr" and pd.notna(hr_grade):
        return f"hr_grade_{int(hr_grade)}"
    if disease_label == "normal":
        return "normal_grade_0"
    return f"{disease_label}_unknown"


def aggregate_groups(frame: pd.DataFrame) -> list[dict]:
    working = frame.copy()
    working["group_anchor"] = working.apply(choose_group_anchor, axis=1)
    working["stratify_key"] = (
        working["source_dataset"].astype(str)
        + "::"
        + working["disease_label"].astype(str)
        + "::"
        + working.apply(severity_bucket, axis=1)
    )

    groups = []
    for group_anchor, group_frame in working.groupby("group_anchor", sort=False):
        groups.append(
            {
                "group_anchor": group_anchor,
                "row_indices": group_frame.index.tolist(),
                "size": len(group_frame),
                "stratify_counts": Counter(group_frame["stratify_key"].tolist()),
            }
        )
    return groups


def greedy_group_split(frame: pd.DataFrame, split_ratio: dict[str, float], seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    groups = aggregate_groups(frame)
    rng.shuffle(groups)
    groups.sort(key=lambda item: (item["size"], len(item["stratify_counts"])), reverse=True)

    split_names = ["train", "val", "test"]
    target_rows = {name: split_ratio[name] * len(frame) for name in split_names}
    stratify_series = (
        frame["source_dataset"].astype(str)
        + "::"
        + frame["disease_label"].astype(str)
        + "::"
        + frame.apply(severity_bucket, axis=1)
    )
    total_stratify = Counter(stratify_series.tolist())
    target_stratify = {
        name: {key: split_ratio[name] * total for key, total in total_stratify.items()}
        for name in split_names
    }

    assignments = {name: [] for name in split_names}
    current_rows = {name: 0 for name in split_names}
    current_stratify = {name: Counter() for name in split_names}

    for group in groups:
        best_split = None
        best_score = None
        for split_name in split_names:
            projected_rows = current_rows[split_name] + group["size"]
            size_penalty = abs(projected_rows - target_rows[split_name]) / max(target_rows[split_name], 1.0)
            stratify_penalty = 0.0
            for key, value in group["stratify_counts"].items():
                projected_value = current_stratify[split_name][key] + value
                target_value = target_stratify[split_name].get(key, 0.0)
                stratify_penalty += abs(projected_value - target_value) / max(target_value, 1.0)
            score = size_penalty + stratify_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assignments[best_split].extend(group["row_indices"])
        current_rows[best_split] += group["size"]
        current_stratify[best_split].update(group["stratify_counts"])

    return assignments


def mark_cross_role_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "quality_flag" not in enriched.columns:
        enriched["quality_flag"] = ""
    duplicate_groups = enriched.loc[enriched["duplicate_group_id"].notna(), ["duplicate_group_id", "dataset_role"]]
    external_groups = set(duplicate_groups.loc[duplicate_groups["dataset_role"] == "external_test", "duplicate_group_id"].tolist())
    if not external_groups:
        return enriched

    for index, row in enriched.iterrows():
        duplicate_group_id = row.get("duplicate_group_id", pd.NA)
        dataset_role = str(row.get("dataset_role", "train") or "train").strip().lower()
        if pd.isna(duplicate_group_id) or dataset_role == "external_test":
            continue
        if duplicate_group_id in external_groups:
            enriched.at[index, "quality_flag"] = append_quality_flag(row.get("quality_flag", ""), "duplicate_with_external_test")
    return enriched


def split_eligible_mask(frame: pd.DataFrame) -> pd.Series:
    dataset_role = frame.get("dataset_role", "train").fillna("train").astype(str).str.strip().str.lower()
    quality_flag = frame.get("quality_flag", "").fillna("").astype(str).str.lower()
    return dataset_role.isin(["", "train"]) & ~quality_flag.str.contains("duplicate_with_external_test", regex=False)


def validate_no_leakage(frame: pd.DataFrame) -> None:
    split_frame = frame[frame["split"].isin(["train", "val", "test"])].copy()
    split_to_patients: dict[str, set[str]] = {}
    for split_name, current_split in split_frame.groupby("split"):
        patients = {patient for patient in current_split["patient_or_case_id"].astype(str) if patient}
        split_to_patients[split_name] = patients

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_to_patients.get(left, set()) & split_to_patients.get(right, set())
        if overlap:
            raise ValueError(f"Patient leakage detected between {left} and {right}: {sorted(overlap)[:10]}")

    split_duplicates = split_frame.loc[split_frame["duplicate_group_id"].notna(), ["split", "duplicate_group_id"]]
    for duplicate_group_id, group in split_duplicates.groupby("duplicate_group_id"):
        if group["split"].nunique() > 1:
            raise ValueError(f"Duplicate leakage detected across splits for group {duplicate_group_id}")


def validate_external_isolation(frame: pd.DataFrame) -> None:
    external_groups = set(
        frame.loc[
            (frame.get("dataset_role", "") == "external_test") & frame["duplicate_group_id"].notna(),
            "duplicate_group_id",
        ].tolist()
    )
    if not external_groups:
        return
    assigned_groups = set(
        frame.loc[
            frame["split"].isin(["train", "val", "test"]) & frame["duplicate_group_id"].notna(),
            "duplicate_group_id",
        ].tolist()
    )
    overlap = sorted(external_groups & assigned_groups)
    if overlap:
        raise ValueError(f"External-test leakage detected for duplicate groups: {overlap[:10]}")


def write_split_files(frame: pd.DataFrame, project_root: Path, config: dict[str, object]) -> None:
    for split_name, relative_path in SPLIT_OUTPUT_PATHS.items():
        output_path = resolve_repo_path(project_root, relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[frame["split"] == split_name].reset_index(drop=True).to_csv(output_path, index=False)

    external_cfg = config.get("external_test", {}) if isinstance(config, dict) else {}
    excluded_flags = {
        str(value).strip().lower()
        for value in (external_cfg.get("exclude_quality_flags", []) if isinstance(external_cfg, dict) else [])
        if str(value).strip()
    }
    external_frame = frame.loc[frame.get("dataset_role", "").astype(str).str.lower() == "external_test"].copy()
    if excluded_flags and not external_frame.empty:
        external_frame = external_frame[
            ~external_frame["quality_flag"].apply(
                lambda value: bool(set(str(value).lower().split("|")) & excluded_flags)
            )
        ]
    output_path = resolve_repo_path(project_root, EXTERNAL_TEST_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_frame.reset_index(drop=True).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(resolve_repo_path(project_root, args.config))
    metadata_path = resolve_repo_path(project_root, args.metadata)
    frame = pd.read_csv(metadata_path)

    if "dataset_role" not in frame.columns:
        frame["dataset_role"] = "train"
    frame["dataset_role"] = frame["dataset_role"].fillna("train").astype(str).str.strip().str.lower()
    if "quality_flag" not in frame.columns:
        frame["quality_flag"] = ""

    frame = enrich_duplicate_groups(frame)
    frame = mark_cross_role_duplicates(frame)

    split_ratio = config.get("splits", {}).get("ratio", {"train": 0.70, "val": 0.15, "test": 0.15})
    seed = int(config.get("splits", {}).get("seed", 42))
    eligible_mask = split_eligible_mask(frame)
    eligible_frame = frame.loc[eligible_mask].copy()
    assignments = greedy_group_split(frame=eligible_frame, split_ratio=split_ratio, seed=seed) if not eligible_frame.empty else {"train": [], "val": [], "test": []}

    frame = frame.copy()
    frame["split"] = ""
    for split_name, indices in assignments.items():
        frame.loc[indices, "split"] = split_name

    validate_no_leakage(frame)
    validate_external_isolation(frame)
    save_metadata(frame, metadata_path)
    write_split_files(frame, project_root, config)
    split_counts = frame.loc[frame["split"].isin(["train", "val", "test"]), "split"].value_counts().to_dict()
    print(f"Created splits: {split_counts}")


if __name__ == "__main__":
    main()
