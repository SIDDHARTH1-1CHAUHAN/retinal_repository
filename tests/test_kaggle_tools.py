import json
import tempfile
import unittest
from pathlib import Path

from src.kaggle.export_bundle import export_bundle
from src.kaggle.import_artifacts import find_repo_artifact_root


class KaggleToolsTests(unittest.TestCase):
    def test_export_bundle_copies_repo_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            repo_root = base / 'wt-integrate'
            (repo_root / 'configs').mkdir(parents=True)
            (repo_root / 'src').mkdir(parents=True)
            (repo_root / 'notebooks').mkdir(parents=True)
            (repo_root / 'tests').mkdir(parents=True)
            (repo_root / 'app.py').write_text('print(1)\n', encoding='utf-8')
            (repo_root / 'README.md').write_text('# demo\n', encoding='utf-8')
            (repo_root / 'requirements.txt').write_text('numpy\n', encoding='utf-8')
            (base / 'test_data').mkdir()

            bundle_root = export_bundle(repo_root, base / 'bundle', include_test_data=True)

            manifest = json.loads((bundle_root / 'bundle_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['repo_dir_name'], 'wt-integrate')
            self.assertTrue((bundle_root / 'wt-integrate' / 'README.md').exists())
            self.assertTrue((bundle_root / 'test_data').exists())

    def test_find_repo_artifact_root_accepts_nested_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir)
            nested_repo = source_root / 'wt-integrate'
            (nested_repo / 'reports').mkdir(parents=True)
            self.assertEqual(find_repo_artifact_root(source_root), nested_repo)


if __name__ == '__main__':
    unittest.main()
