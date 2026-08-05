import base64
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from services import image_restorer  # noqa: E402


class _FakeImages:
    def __init__(self, output_image):
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        self.encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        self.kwargs = None

    def edit(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(data=[SimpleNamespace(b64_json=self.encoded)])


class _FakeClient:
    def __init__(self, output_image):
        self.images = _FakeImages(output_image)


class ImageRestorerTests(unittest.TestCase):
    def config(self):
        return image_restorer.RestorationConfig(
            model="gpt-image-2",
            quality="medium",
            size="auto",
            timeout=30,
            mask_blur_radius=0,
            side_pad_ratio=0,
            top_pad_ratio=0,
            bottom_pad_ratio=0,
            watermark_fraction=0,
        )

    def test_disjoint_boxes_do_not_mask_the_space_between_them(self):
        mask = image_restorer.build_damage_mask(
            (100, 100),
            [(10, 10, 20, 20), (70, 70, 80, 80)],
            self.config(),
        )
        self.assertEqual(mask.getpixel((15, 15)), 255)
        self.assertEqual(mask.getpixel((75, 75)), 255)
        self.assertEqual(mask.getpixel((50, 50)), 0)

    def test_openai_mask_uses_transparency_for_edit_region(self):
        edit_mask = image_restorer.build_damage_mask(
            (100, 100), [(10, 10, 20, 20)], self.config()
        )
        api_mask = image_restorer.build_openai_mask(edit_mask)
        self.assertEqual(api_mask.getpixel((15, 15))[3], 0)
        self.assertEqual(api_mask.getpixel((50, 50))[3], 255)

    def test_restore_calls_openai_and_preserves_pixels_outside_mask(self):
        original = Image.new("RGB", (100, 100), "red")
        generated = Image.new("RGB", (100, 100), "green")
        client = _FakeClient(generated)
        detections = [
            {
                "part": "앞범퍼",
                "part_en": "front-bumper-dent",
                "damage_type": "찌그러짐",
                "damage_type_en": "dent",
                "bbox": [10, 10, 20, 20],
            }
        ]

        result = image_restorer.restore_image(
            original,
            detections,
            client=client,
            config=self.config(),
        )

        self.assertEqual(result.getpixel((15, 15)), (0, 128, 0))
        self.assertEqual(result.getpixel((50, 50)), (255, 0, 0))
        request = client.images.kwargs
        self.assertEqual(request["model"], "gpt-image-2")
        self.assertEqual(request["n"], 1)
        self.assertNotIn("input_fidelity", request)

        sent_mask = Image.open(io.BytesIO(request["mask"][1])).convert("RGBA")
        self.assertEqual(request["size"], "1024x1024")
        self.assertEqual(sent_mask.size, (1024, 1024))
        self.assertEqual(sent_mask.getpixel((150, 150))[3], 0)
        self.assertEqual(sent_mask.getpixel((500, 500))[3], 255)

    def test_landscape_input_uses_landscape_canvas_without_stretching_result(self):
        original = Image.new("RGB", (200, 100), "red")
        generated = Image.new("RGB", (1536, 1024), "green")
        client = _FakeClient(generated)
        detections = [{"bbox": [20, 20, 40, 40]}]

        result = image_restorer.restore_image(
            original,
            detections,
            client=client,
            config=self.config(),
        )

        self.assertEqual(result.size, original.size)
        self.assertEqual(result.getpixel((30, 30)), (0, 128, 0))
        self.assertEqual(result.getpixel((100, 50)), (255, 0, 0))
        request = client.images.kwargs
        self.assertEqual(request["size"], "1536x1024")
        sent_image = Image.open(io.BytesIO(request["image"][1]))
        self.assertEqual(sent_image.size, (1536, 1024))


if __name__ == "__main__":
    unittest.main()
