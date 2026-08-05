import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from routers import repair_preview  # noqa: E402


def _image_bytes(size=(80, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format="PNG")
    return buffer.getvalue()


DETECTIONS = [
    {
        "part": "front bumper",
        "part_en": "front-bumper-dent",
        "damage_type": "dent",
        "damage_type_en": "dent",
        "severity": "minor",
        "confidence": 0.9,
        "bbox": [10, 10, 30, 30],
    }
]


class RepairPreviewRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(repair_preview.router)
        cls.client = TestClient(app)

    def setUp(self):
        repair_preview._request_times.clear()

    def _post(self, headers=None):
        return self.client.post(
            "/repair-preview",
            files={"file": ("vehicle.png", _image_bytes(), "image/png")},
            data={"detections_json": json.dumps(DETECTIONS)},
            headers=headers or {},
        )

    def test_shared_token_is_required_when_configured(self):
        with patch.object(repair_preview, "REPAIR_API_TOKEN", "internal-secret"):
            response = self._post()
        self.assertEqual(response.status_code, 401)

    def test_success_does_not_require_openai_in_unit_test(self):
        with (
            patch.object(repair_preview, "REPAIR_API_TOKEN", "internal-secret"),
            patch.object(
                repair_preview.image_restorer,
                "restore_image",
                return_value=Image.new("RGB", (80, 60), "green"),
            ),
        ):
            response = self._post({"X-Repair-Token": "internal-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()
