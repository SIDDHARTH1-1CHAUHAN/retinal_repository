import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess_images import preprocess_image
from src.inference.benchmark_demo import (
    BenchmarkLookupService,
    build_benchmark_report,
    severity_index_100,
    severity_label_for_grade,
)


def _write_rgb_image(path: Path, array: np.ndarray) -> None:
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


class BenchmarkDemoTests(unittest.TestCase):
    def test_severity_helpers(self) -> None:
        self.assertEqual(severity_label_for_grade(2), "Moderate NPDR (Grade 2)")
        self.assertEqual(severity_index_100(4), 100.0)

    def test_benchmark_lookup_matches_exact_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_a = np.full((40, 40, 3), 120, dtype=np.uint8)
            image_b = np.full((40, 40, 3), 220, dtype=np.uint8)
            path_a = root / "a.png"
            path_b = root / "b.png"
            _write_rgb_image(path_a, image_a)
            _write_rgb_image(path_b, image_b)

            frame = pd.DataFrame(
                [
                    {
                        "image_id": "case_a",
                        "source_dataset": "clinical_dr_test",
                        "patient_or_case_id": "case_a",
                        "image_path": str(path_a),
                        "raw_image_path": str(path_a),
                        "disease_label": "normal",
                        "dr_grade": 0,
                        "hr_grade": pd.NA,
                        "split": "",
                        "is_manual_label": True,
                        "dataset_role": "external_test",
                        "quality_flag": "",
                    },
                    {
                        "image_id": "case_b",
                        "source_dataset": "clinical_dr_test",
                        "patient_or_case_id": "case_b",
                        "image_path": str(path_b),
                        "raw_image_path": str(path_b),
                        "disease_label": "dr",
                        "dr_grade": 3,
                        "hr_grade": pd.NA,
                        "split": "",
                        "is_manual_label": True,
                        "dataset_role": "external_test",
                        "quality_flag": "",
                    },
                ]
            )
            settings = {
                "crop_black_borders": {"enabled": False},
                "gaussian_blur": {"enabled": False},
                "clahe": {"enabled": False},
                "resize": {"size": [224, 224]},
                "normalize": {"enabled": True, "scale": 255.0},
            }
            service = BenchmarkLookupService(frame, settings)
            tensor_b = preprocess_image(path_b, settings)
            prediction = service.predict(tensor_b)
            self.assertEqual(prediction["status"], "matched_benchmark")
            self.assertEqual(prediction["grade"], 3)
            self.assertEqual(prediction["severity"], "Severe NPDR (Grade 3)")
            self.assertEqual(prediction["dr_possibility"], 1.0)
            self.assertIsNone(prediction["hr_possibility"])

            unknown = np.zeros((224, 224, 3), dtype=np.float32)
            unknown_prediction = service.predict(unknown)
            self.assertEqual(unknown_prediction["status"], "unvalidated_input")

    def test_benchmark_report_excludes_ambiguous_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {"dr_grade": 0, "quality_flag": ""},
                {"dr_grade": 2, "quality_flag": ""},
                {"dr_grade": 0, "quality_flag": "ambiguous_stage0_with_lesions"},
            ]
        )
        report = build_benchmark_report(frame)
        self.assertEqual(report["total_rows"], 3)
        self.assertEqual(report["display_rows"], 2)
        self.assertEqual(report["excluded_rows"], 1)
        self.assertIn("ambiguous_stage0_with_lesions", report["excluded_quality_counts"])


if __name__ == "__main__":
    unittest.main()
