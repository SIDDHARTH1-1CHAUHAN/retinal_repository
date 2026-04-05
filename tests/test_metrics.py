import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import evaluate_predictions


class EvaluatePredictionsTest(unittest.TestCase):
    def test_binary_metrics_and_confidence_fields(self) -> None:
        probabilities = [
            [0.9, 0.1],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.2, 0.8],
        ]
        report = evaluate_predictions(
            y_true=[0, 1, 1, 0],
            probabilities=probabilities,
            label_names=["normal", "dr"],
            positive_labels=["dr"],
            ordered=True,
        )

        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(report["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(report["mean_confidence"], 0.75)
        self.assertIn("ece", report)
        self.assertIn("brier_score", report)
        self.assertIn("quadratic_weighted_kappa", report)
        self.assertEqual(report["confusion_matrix"], [[1, 1], [1, 1]])

        binary_view = report["binary_views"][0]
        self.assertEqual(binary_view["label"], "dr")
        self.assertAlmostEqual(binary_view["sensitivity"], 0.5)
        self.assertAlmostEqual(binary_view["specificity"], 0.5)

    def test_requires_prediction_source(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions(y_true=[0, 1])


if __name__ == "__main__":
    unittest.main()
