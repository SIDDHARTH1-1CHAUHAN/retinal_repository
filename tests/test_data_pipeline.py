import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_master_metadata import choose_clinical_fundus_image, parse_clinical_diagnosis_text
from src.data.contracts import ensure_metadata_contract, is_stage2_dr_eligible_row


class DataPipelineContractTest(unittest.TestCase):
    def test_contract_accepts_normal_grade_zero_and_external_test_role(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "image_id": "normal_1",
                    "source_dataset": "clinical",
                    "patient_or_case_id": "case_1",
                    "image_path": "sample_a.jpg",
                    "disease_label": "normal",
                    "dr_grade": 0,
                    "hr_grade": pd.NA,
                    "split": "",
                    "is_manual_label": True,
                    "dataset_role": "external_test",
                },
                {
                    "image_id": "dr_1",
                    "source_dataset": "eyepacs",
                    "patient_or_case_id": "case_2",
                    "image_path": "sample_b.jpg",
                    "disease_label": "dr",
                    "dr_grade": 2,
                    "hr_grade": pd.NA,
                    "split": "train",
                    "is_manual_label": True,
                    "dataset_role": "train",
                },
            ]
        )

        normalized = ensure_metadata_contract(frame)
        normal_row = normalized.loc[normalized["image_id"] == "normal_1"].iloc[0]
        dr_row = normalized.loc[normalized["image_id"] == "dr_1"].iloc[0]
        self.assertEqual(normal_row["dr_grade"], 0)
        self.assertEqual(normal_row["dataset_role"], "external_test")
        self.assertTrue(is_stage2_dr_eligible_row(dr_row))
        self.assertFalse(is_stage2_dr_eligible_row(normal_row))

    def test_parse_clinical_diagnosis_text_handles_stage_and_proliferative(self) -> None:
        self.assertEqual(parse_clinical_diagnosis_text("Non proliferative Diabetic Retinopathy stage 3")[0], 3)
        self.assertEqual(parse_clinical_diagnosis_text("Onset proliferative Diabetic Retinopathy")[0], 4)
        self.assertEqual(parse_clinical_diagnosis_text("No DR Lesions Present")[0], 0)

    def test_choose_clinical_fundus_image_skips_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            (case_dir / "Im10.jpg").write_bytes(b"fundus")
            (case_dir / "Im10_OD_GT.jpg").write_bytes(b"mask")
            (case_dir / "Im10_Hemorhages_GT.jpg").write_bytes(b"mask")
            chosen = choose_clinical_fundus_image(case_dir)
            self.assertIsNotNone(chosen)
            self.assertEqual(chosen.name, "Im10.jpg")


if __name__ == "__main__":
    unittest.main()
