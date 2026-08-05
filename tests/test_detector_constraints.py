import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from services import detector  # noqa: E402


class _FixedClassifier:
    def __init__(self, logits):
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, _tensor):
        return self._logits


class DetectorConstraintTests(unittest.TestCase):
    def test_part_constraints_follow_physical_damage_domain(self):
        self.assertEqual(
            detector.PART_VALID_DAMAGES["front-bumper-dent"],
            {"scratch", "dent", "crack"},
        )
        self.assertEqual(
            detector.PART_VALID_DAMAGES["Front-Windscreen-Damage"],
            {"crack", "glass shatter"},
        )
        self.assertEqual(
            detector.PART_VALID_DAMAGES["Headlight-Damage"],
            {"crack", "lamp broken"},
        )

    def test_classifier_masks_impossible_damage_for_detected_part(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        classes = ["lamp broken", "scratch", "dent"]
        classifier = _FixedClassifier([10.0, 8.0, 7.0])

        with (
            patch.object(detector, "_damage_type_clf", classifier),
            patch.object(detector, "_damage_type_classes", classes),
        ):
            constrained = detector._classify_damage_type(
                image,
                (2, 2, 18, 18),
                "front-bumper-dent",
            )
            unconstrained = detector._classify_damage_type(
                image,
                (2, 2, 18, 18),
                "unknown-part",
            )

        self.assertEqual(constrained, "scratch")
        self.assertEqual(unconstrained, "lamp broken")


if __name__ == "__main__":
    unittest.main()
