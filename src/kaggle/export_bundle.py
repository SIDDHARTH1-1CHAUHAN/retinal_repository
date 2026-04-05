from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO_EXPORT_RELATIVE_PATHS = (
    'configs',
    'src',
    'notebooks',
    'tests',
    'app.py',
    'README.md',
    'requirements.txt',
)
IGNORE_NAMES = {'__pycache__', '.git', '.pytest_cache'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export a clean Kaggle training bundle.')
    parser.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--output-root', default='artifacts/kaggle_bundle')
    parser.add_argument('--include-test-data', action='store_true')
    return parser.parse_args()


def _copy_entry(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_NAMES, '*.pyc', '*.pyo'),
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_bundle_manifest(repo_root: Path, include_test_data: bool) -> dict[str, Any]:
    return {
        'repo_dir_name': repo_root.name,
        'entry_script': f'{repo_root.name}/notebooks/kaggle_train.py',
        'include_test_data': include_test_data,
        'expected_training_inputs': {
            'dr_combined': 'https://www.kaggle.com/datasets/harsha1289/combined-dr-dataset-aptosidridmessidoreyepacs',
            'eyepacs': 'https://www.kaggle.com/datasets/dreamer07/eyepacs',
            'odir': 'attach_public_odir_dataset',
            'rvm': 'required_private_or_uploaded_rvm_dataset',
        },
        'run_sequence': [
            'python src/data/build_master_metadata.py --config configs/data/data_config.yaml',
            'python src/data/preprocess_images.py --config configs/data/data_config.yaml',
            'python src/data/make_splits.py --config configs/data/data_config.yaml',
            'python src/training/train_stage1.py --config configs/model_stage1.yaml',
            'python src/training/train_stage2_dr.py --config configs/model_stage2.yaml',
            'python src/training/train_stage2_hr.py --config configs/model_stage2.yaml',
        ],
    }


def export_bundle(repo_root: Path, output_root: Path, include_test_data: bool = False) -> Path:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    export_repo_root = output_root / repo_root.name

    if output_root.exists():
        shutil.rmtree(output_root)
    export_repo_root.parent.mkdir(parents=True, exist_ok=True)

    for relative_path in REPO_EXPORT_RELATIVE_PATHS:
        source = repo_root / relative_path
        if not source.exists():
            continue
        _copy_entry(source, export_repo_root / relative_path)

    if include_test_data:
        local_test_data = repo_root.parent / 'test_data'
        if local_test_data.exists():
            _copy_entry(local_test_data, output_root / 'test_data')

    manifest = build_bundle_manifest(repo_root=repo_root, include_test_data=include_test_data)
    (output_root / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output_root


def main() -> None:
    args = parse_args()
    bundle_root = export_bundle(
        repo_root=Path(args.repo_root),
        output_root=Path(args.output_root),
        include_test_data=bool(args.include_test_data),
    )
    print(json.dumps({'bundle_root': str(bundle_root)}, indent=2))


if __name__ == '__main__':
    main()
