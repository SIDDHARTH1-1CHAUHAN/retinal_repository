from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

COPY_RELATIVE_PATHS = (
    'reports',
    'data/metadata',
    'data/splits',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Import trained Kaggle artifacts into the local repo.')
    parser.add_argument('--source', required=True, help='Directory or zip file produced by the Kaggle run.')
    parser.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[2]))
    return parser.parse_args()


def _copy_entry(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def find_repo_artifact_root(source_root: Path) -> Path:
    candidates = [source_root]
    if source_root.exists() and source_root.is_dir():
        candidates.extend(path for path in source_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / 'reports').exists():
            return candidate
    raise FileNotFoundError(f'Could not locate a repo-like artifact root under {source_root}')


def _materialize_source(source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if source.is_file() and source.suffix.lower() == '.zip':
        temp_dir = tempfile.TemporaryDirectory(prefix='kaggle_artifacts_')
        with zipfile.ZipFile(source, 'r') as archive:
            archive.extractall(temp_dir.name)
        return Path(temp_dir.name), temp_dir
    return source, None


def import_artifacts(source: Path, repo_root: Path) -> list[str]:
    source = source.resolve()
    repo_root = repo_root.resolve()
    materialized_source, temp_dir = _materialize_source(source)
    try:
        artifact_root = find_repo_artifact_root(materialized_source)
        copied: list[str] = []
        for relative_path in COPY_RELATIVE_PATHS:
            source_path = artifact_root / relative_path
            if not source_path.exists():
                continue
            destination_path = repo_root / relative_path
            _copy_entry(source_path, destination_path)
            copied.append(relative_path)
        if not copied:
            raise FileNotFoundError(f'No importable artifact directories were found under {artifact_root}')
        return copied
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> None:
    args = parse_args()
    copied = import_artifacts(source=Path(args.source), repo_root=Path(args.repo_root))
    print(json.dumps({'copied': copied}, indent=2))


if __name__ == '__main__':
    main()
