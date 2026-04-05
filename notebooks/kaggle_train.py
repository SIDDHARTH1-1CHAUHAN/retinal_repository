from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

RAW_LAYOUT = {
    'aptos_csv': 'data/raw/aptos/train.csv',
    'aptos_images': 'data/raw/aptos/images',
    'messidor_csv': 'data/raw/messidor/labels.csv',
    'messidor_images': 'data/raw/messidor/images',
    'idrid_csv': 'data/raw/idrid/labels.csv',
    'idrid_images': 'data/raw/idrid/images',
    'eyepacs_csv': 'data/raw/eyepacs/trainLabels.csv',
    'eyepacs_images': 'data/raw/eyepacs/images',
    'odir_csv': 'data/raw/odir/full_df.csv',
    'odir_images': 'data/raw/odir/images',
    'rvm_csv': 'data/raw/rvm/labels.csv',
    'rvm_images': 'data/raw/rvm/images',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the retinal pipeline on Kaggle GPU.')
    parser.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--combined-dr-root')
    parser.add_argument('--eyepacs-root')
    parser.add_argument('--odir-root')
    parser.add_argument('--rvm-root')
    parser.add_argument('--output-root', default='/kaggle/working/retinal_training_output')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--skip-preprocess', action='store_true')
    parser.add_argument('--skip-stage2-hr', action='store_true')
    return parser.parse_args()


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        remove_existing(destination)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def find_file(root: Path, name_hints: list[str], keywords: list[str]) -> Path:
    lowered_hints = {value.lower() for value in name_hints}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        haystack = str(path.relative_to(root)).lower()
        if path.name.lower() in lowered_hints:
            if all(keyword in haystack for keyword in keywords):
                return path
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        haystack = str(path.relative_to(root)).lower()
        if all(keyword in haystack for keyword in keywords):
            return path
    raise FileNotFoundError(f'Could not find file in {root} matching hints={name_hints} keywords={keywords}')


def find_dir(root: Path, name_hints: list[str], keywords: list[str]) -> Path:
    lowered_hints = {value.lower() for value in name_hints}
    for path in root.rglob('*'):
        if not path.is_dir():
            continue
        haystack = str(path.relative_to(root)).lower()
        if path.name.lower() in lowered_hints and all(keyword in haystack for keyword in keywords):
            return path
    for path in root.rglob('*'):
        if not path.is_dir():
            continue
        haystack = str(path.relative_to(root)).lower()
        if all(keyword in haystack for keyword in keywords):
            return path
    raise FileNotFoundError(f'Could not find directory in {root} matching hints={name_hints} keywords={keywords}')


def stage_combined_dr_dataset(repo_root: Path, combined_root: Path) -> None:
    mapping = [
        ('aptos', ['train.csv'], ['aptos'], ['images', 'train_images'], ['aptos']),
        ('messidor', ['labels.csv', 'train.csv'], ['messidor'], ['images', 'train_images'], ['messidor']),
        ('idrid', ['labels.csv', 'train.csv'], ['idrid'], ['images', 'train_images'], ['idrid']),
    ]
    for dataset_name, file_hints, file_keywords, dir_hints, dir_keywords in mapping:
        csv_source = find_file(combined_root, file_hints, file_keywords)
        image_source = find_dir(combined_root, dir_hints, dir_keywords)
        link_or_copy(csv_source, repo_root / RAW_LAYOUT[f'{dataset_name}_csv'])
        link_or_copy(image_source, repo_root / RAW_LAYOUT[f'{dataset_name}_images'])


def stage_single_dataset(repo_root: Path, source_root: Path, csv_target_key: str, image_target_key: str, file_hints: list[str], file_keywords: list[str], dir_hints: list[str], dir_keywords: list[str]) -> None:
    csv_source = find_file(source_root, file_hints, file_keywords)
    image_source = find_dir(source_root, dir_hints, dir_keywords)
    link_or_copy(csv_source, repo_root / RAW_LAYOUT[csv_target_key])
    link_or_copy(image_source, repo_root / RAW_LAYOUT[image_target_key])


def run_command(command: list[str], cwd: Path) -> None:
    print('+', ' '.join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def copy_output_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def package_outputs(repo_root: Path, output_root: Path) -> None:
    export_repo_root = output_root / repo_root.name
    export_repo_root.mkdir(parents=True, exist_ok=True)
    for relative_path in ('reports', 'data/metadata', 'data/splits'):
        source = repo_root / relative_path
        if source.exists():
            copy_output_tree(source, export_repo_root / relative_path)
    manifest = {
        'repo_dir_name': repo_root.name,
        'copied_paths': ['reports', 'data/metadata', 'data/splits'],
    }
    (output_root / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()

    if args.combined_dr_root:
        stage_combined_dr_dataset(repo_root, Path(args.combined_dr_root).resolve())
    if args.eyepacs_root:
        stage_single_dataset(
            repo_root,
            Path(args.eyepacs_root).resolve(),
            'eyepacs_csv',
            'eyepacs_images',
            ['trainLabels.csv', 'labels.csv'],
            ['eyepacs'],
            ['images', 'train'],
            ['eyepacs'],
        )
    if args.odir_root:
        stage_single_dataset(
            repo_root,
            Path(args.odir_root).resolve(),
            'odir_csv',
            'odir_images',
            ['full_df.csv', 'labels.csv'],
            ['odir'],
            ['images', 'fundus images', 'fundus_images'],
            ['odir'],
        )
    if args.rvm_root:
        stage_single_dataset(
            repo_root,
            Path(args.rvm_root).resolve(),
            'rvm_csv',
            'rvm_images',
            ['labels.csv', 'train.csv'],
            ['rvm'],
            ['images', 'fundus_images'],
            ['rvm'],
        )

    if not args.skip_stage2_hr and not (repo_root / RAW_LAYOUT['rvm_csv']).exists():
        raise FileNotFoundError('RVM is required for Stage 2 HR training. Attach it and pass --rvm-root.')

    commands = [
        [args.python_bin, 'src/data/build_master_metadata.py', '--config', 'configs/data/data_config.yaml'],
        [args.python_bin, 'src/data/make_splits.py', '--config', 'configs/data/data_config.yaml'],
        [args.python_bin, 'src/training/train_stage1.py', '--config', 'configs/model_stage1.yaml'],
        [args.python_bin, 'src/training/train_stage2_dr.py', '--config', 'configs/model_stage2.yaml'],
    ]
    if not args.skip_preprocess:
        commands.insert(1, [args.python_bin, 'src/data/preprocess_images.py', '--config', 'configs/data/data_config.yaml'])
    if not args.skip_stage2_hr:
        commands.append([args.python_bin, 'src/training/train_stage2_hr.py', '--config', 'configs/model_stage2.yaml'])

    for command in commands:
        run_command(command, cwd=repo_root)

    package_outputs(repo_root=repo_root, output_root=output_root)
    print(json.dumps({'artifact_root': str(output_root)}, indent=2))


if __name__ == '__main__':
    main()
