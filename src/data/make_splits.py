from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.contracts import (
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


def dhash_from_array(array: np.ndarray) -> str:
    cv2 = require_cv2()
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if bit else "0" for bit in diff.flatten())
    return f"{int(bits, 2):016x}"


def enrich_duplicate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    cv2 = require_cv2()
    if "duplicate_group_id" in frame.columns and frame["duplicate_group_id"].notna().any():
        return frame

    duplicate_map: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        image_path = Path(row["image_path"])
        if image_path.suffix.lower() == ".npy" and image_path.exists():
            hash_key = dhash_from_array(load_image_tensor(image_path))
        elif "raw_image_path" in frame.columns and Path(row["raw_image_path"]).exists():
            raw_bgr = cv2.imread(str(row["raw_image_path"]), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                continue
            hash_key = dhash_from_array(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        else:
            continue
        duplicate_map[hash_key].append(index)

    frame = frame.copy()
    frame["duplicate_group_id"] = pd.NA
    for hash_key, indices in duplicate_map.items():
        if len(indices) >= 2:
            frame.loc[indices, "duplicate_group_id"] = f"dup_{hash_key}"
    return frame


def choose_group_anchor(row: pd.Series) -> str:
    duplicate_group_id = row.get("duplicate_group_id", pd.NA)
    if pd.notna(duplicate_group_id):
        return str(duplicate_group_id)
    patient_id = str(row["patient_or_case_id"]).strip()
    if patient_id:
        return patient_id
    return str(row["image_id"])


def aggregate_groups(frame: pd.DataFrame) -> list[dict]:
    working = frame.copy()
    working["group_anchor"] = working.apply(choose_group_anchor, axis=1)
    working["stratify_key"] = working["source_dataset"].astype(str) + "::" + working["disease_label"].astype(str)

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
    total_stratify = Counter(frame["source_dataset"].astype(str) + "::" + frame["disease_label"].astype(str))
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


def validate_no_leakage(frame: pd.DataFrame) -> None:
    split_to_patients: dict[str, set[str]] = {}
    for split_name, split_frame in frame.groupby("split"):
        patients = {patient for patient in split_frame["patient_or_case_id"].astype(str) if patient}
        split_to_patients[split_name] = patients

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_to_patients.get(left, set()) & split_to_patients.get(right, set())
        if overlap:
            raise ValueError(f"Patient leakage detected between {left} and {right}: {sorted(overlap)[:10]}")


def write_split_files(frame: pd.DataFrame, project_root: Path) -> None:
    for split_name, relative_path in SPLIT_OUTPUT_PATHS.items():
        output_path = resolve_repo_path(project_root, relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[frame["split"] == split_name].reset_index(drop=True).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(resolve_repo_path(project_root, args.config))
    metadata_path = resolve_repo_path(project_root, args.metadata)
    frame = pd.read_csv(metadata_path)

    frame = enrich_duplicate_groups(frame)
    split_ratio = config.get("splits", {}).get("ratio", {"train": 0.70, "val": 0.15, "test": 0.15})
    seed = int(config.get("splits", {}).get("seed", 42))
    assignments = greedy_group_split(frame=frame, split_ratio=split_ratio, seed=seed)

    frame = frame.copy()
    frame["split"] = ""
    for split_name, indices in assignments.items():
        frame.loc[indices, "split"] = split_name

    validate_no_leakage(frame)
    save_metadata(frame, metadata_path)
    write_split_files(frame, project_root)
    print(f"Created splits: {frame['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
