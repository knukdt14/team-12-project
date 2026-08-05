"""OpenAI Image API를 이용한 차량 손상 복원 시뮬레이션.

원본과 YOLO 탐지 박스로 편집 마스크를 만든 뒤 OpenAI에 한 장의 이미지 편집을
요청한다. 모델이 마스크 바깥을 바꾸더라도 최종 단계에서 원본과 다시 합성해
정상 영역의 픽셀을 보존한다.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError


class RestorationError(RuntimeError):
    """복원 요청을 안전한 사용자 메시지로 변환할 수 있는 기본 예외."""

    status_code = 502
    public_message = "AI 복원 서비스에서 정상적인 결과를 받지 못했습니다."


class RestorationNotConfigured(RestorationError):
    status_code = 503
    public_message = "OpenAI API 키가 설정되지 않아 복원 이미지를 만들 수 없습니다."


class InvalidRestorationInput(RestorationError):
    status_code = 422
    public_message = "복원할 손상 영역 정보가 올바르지 않습니다."


class RestorationProviderError(RestorationError):
    def __init__(self, public_message: str, status_code: int = 502):
        super().__init__(public_message)
        self.public_message = public_message
        self.status_code = status_code


@dataclass(frozen=True)
class RestorationConfig:
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip()
    )
    quality: str = field(
        default_factory=lambda: os.getenv("OPENAI_IMAGE_QUALITY", "medium").strip()
    )
    size: str = field(
        default_factory=lambda: os.getenv("OPENAI_IMAGE_SIZE", "auto").strip()
    )
    timeout: float = field(
        default_factory=lambda: float(os.getenv("OPENAI_IMAGE_TIMEOUT", "180"))
    )
    mask_blur_radius: int = 14
    side_pad_ratio: float = 0.18
    top_pad_ratio: float = 0.12
    bottom_pad_ratio: float = 0.35
    watermark_fraction: float = field(
        default_factory=lambda: float(os.getenv("REPAIR_WATERMARK_FRACTION", "0"))
    )


def status() -> dict:
    return {
        "provider": "openai",
        "configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip(),
    }


@lru_cache(maxsize=2)
def _client_for_key(api_key: str, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RestorationProviderError(
            "OpenAI SDK가 설치되지 않아 복원 서비스를 시작할 수 없습니다.",
            status_code=503,
        ) from exc
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=1)


def _clip_box(box: Sequence[float], width: int, height: int) -> tuple[float, ...]:
    if width <= 0 or height <= 0 or len(box) != 4:
        raise InvalidRestorationInput()
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise InvalidRestorationInput() from exc

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise InvalidRestorationInput()
    if x2 <= x1 or y2 <= y1:
        raise InvalidRestorationInput()

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise InvalidRestorationInput()
    return x1, y1, x2, y2


def build_damage_mask(
    image_size: tuple[int, int],
    boxes: Iterable[Sequence[float]],
    config: RestorationConfig | None = None,
) -> Image.Image:
    """각 탐지 박스의 합집합을 흰색(편집 대상)으로 표시한 L 마스크를 만든다."""
    config = config or RestorationConfig()
    width, height = image_size
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    count = 0

    for box in boxes:
        x1, y1, x2, y2 = _clip_box(box, width, height)
        box_width, box_height = x2 - x1, y2 - y1
        draw.rectangle(
            [
                max(0, int(x1 - box_width * config.side_pad_ratio)),
                max(0, int(y1 - box_height * config.top_pad_ratio)),
                min(width, int(x2 + box_width * config.side_pad_ratio)),
                min(height, int(y2 + box_height * config.bottom_pad_ratio)),
            ],
            fill=255,
        )
        count += 1

    if count == 0 or mask.getbbox() is None:
        raise InvalidRestorationInput()

    # 수업용 샘플의 우측 하단 워터마크가 깨진 글자로 재생성되는 것을 방지한다.
    fraction = max(0.0, min(0.4, config.watermark_fraction))
    if fraction:
        draw.rectangle(
            [width * (1 - fraction), height * (1 - fraction), width, height],
            fill=0,
        )
    if mask.getbbox() is None:
        raise InvalidRestorationInput()
    return mask


def build_openai_mask(edit_mask: Image.Image) -> Image.Image:
    """흰색=편집인 내부 마스크를 OpenAI의 투명 영역=편집 RGBA 마스크로 변환."""
    binary = edit_mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
    rgba = Image.new("RGBA", binary.size, (255, 255, 255, 255))
    rgba.putalpha(ImageOps.invert(binary))
    return rgba


def _resolve_canvas_size(
    requested_size: str,
    image_size: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    supported = {
        "1024x1024": (1024, 1024),
        "1536x1024": (1536, 1024),
        "1024x1536": (1024, 1536),
    }
    if requested_size != "auto":
        if requested_size not in supported:
            raise InvalidRestorationInput()
        return requested_size, supported[requested_size]

    width, height = image_size
    ratio = width / height
    if ratio >= 1.2:
        return "1536x1024", supported["1536x1024"]
    if ratio <= 1 / 1.2:
        return "1024x1536", supported["1024x1536"]
    return "1024x1024", supported["1024x1024"]


def _prepare_api_canvas(
    image: Image.Image,
    edit_mask: Image.Image,
    requested_size: str,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], str]:
    """원본 비율을 유지해 지원 캔버스에 넣고 결과를 되돌릴 crop 좌표를 반환."""
    output_size, canvas_size = _resolve_canvas_size(requested_size, image.size)
    canvas_width, canvas_height = canvas_size
    scale = min(canvas_width / image.width, canvas_height / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized_image = image.resize(resized_size, Image.Resampling.LANCZOS)
    resized_mask = edit_mask.resize(resized_size, Image.Resampling.NEAREST)
    left = (canvas_width - resized_size[0]) // 2
    top = (canvas_height - resized_size[1]) // 2
    content_box = (left, top, left + resized_size[0], top + resized_size[1])

    background = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    canvas = Image.new("RGB", canvas_size, background)
    canvas.paste(resized_image, (left, top))
    canvas_mask = Image.new("L", canvas_size, 0)
    canvas_mask.paste(resized_mask, (left, top))
    return canvas, canvas_mask, content_box, output_size


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _build_prompt(detections: Sequence[Mapping[str, object]]) -> str:
    labels = []
    for detection in detections:
        part = str(detection.get("part") or detection.get("part_en") or "차체 부위")
        damage = str(
            detection.get("damage_type") or detection.get("damage_type_en") or "외관 손상"
        )
        labels.append(f"{part}: {damage}")
    damaged_parts = ", ".join(dict.fromkeys(labels))

    return (
        "Create a photorealistic post-repair version of this exact vehicle photo. "
        f"The transparent mask identifies collision damage to repair ({damaged_parts}). "
        "Within the masked areas, restore the original factory body geometry and finish: "
        "remove dents, cracks, scratches, broken pieces and hanging debris; reconstruct only "
        "the damaged OEM panels, lamps, glass or bumper parts as intact and professionally "
        "repaired. Preserve the exact same vehicle identity, manufacturer design, paint color, "
        "panel gaps, badges, grille pattern, wheels and license plate. Preserve the camera "
        "position, crop, perspective, lighting, reflections, ground, background, people and "
        "other vehicles. Do not customize the car, add objects, change text, or alter any "
        "unmasked area. The output must look like the same photograph taken after repair."
    )


def _provider_error(exc: Exception) -> RestorationProviderError:
    name = type(exc).__name__
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return RestorationProviderError(
            "OpenAI API 키 또는 이미지 모델 사용 권한을 확인해주세요.", 503
        )
    if name == "RateLimitError":
        return RestorationProviderError(
            "OpenAI 요청 한도에 도달했습니다. 잠시 후 다시 시도해주세요.", 429
        )
    if name in {"APITimeoutError", "TimeoutError"}:
        return RestorationProviderError(
            "복원 이미지 생성 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.", 504
        )
    if name == "BadRequestError":
        return RestorationProviderError(
            "OpenAI가 이 이미지 편집 요청을 처리하지 못했습니다. 다른 사진으로 시도해주세요.",
            422,
        )
    if name == "APIConnectionError":
        return RestorationProviderError(
            "OpenAI 복원 서비스에 연결하지 못했습니다.", 502
        )
    return RestorationProviderError(
        "OpenAI 복원 서비스에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        502,
    )


def restore_image(
    original_image: Image.Image,
    detections: Sequence[Mapping[str, object]],
    *,
    client=None,
    config: RestorationConfig | None = None,
) -> Image.Image:
    """OpenAI 편집 결과를 손상 마스크 안에만 합성해 원본 크기의 이미지를 반환."""
    config = config or RestorationConfig()
    if config.quality not in {"low", "medium", "high", "auto"}:
        raise InvalidRestorationInput()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if client is None:
        if not api_key:
            raise RestorationNotConfigured()
        client = _client_for_key(api_key, config.timeout)

    # /diagnose의 OpenCV 디코더가 보는 원시 픽셀 좌표와 박스를 일치시킨다.
    # 여기서만 EXIF 회전을 적용하면 휴대폰 사진의 마스크 위치가 어긋난다.
    original = original_image.convert("RGB")
    boxes = [detection.get("bbox", []) for detection in detections]
    edit_mask = build_damage_mask(original.size, boxes, config)

    api_image, api_edit_mask, content_box, output_size = _prepare_api_canvas(
        original,
        edit_mask,
        config.size,
    )
    api_mask = build_openai_mask(api_edit_mask)
    image_bytes = _png_bytes(api_image)
    mask_bytes = _png_bytes(api_mask)

    if len(image_bytes) >= 50 * 1024 * 1024:
        raise InvalidRestorationInput()

    request = {
        "model": config.model,
        "image": ("vehicle.png", image_bytes, "image/png"),
        "mask": ("damage-mask.png", mask_bytes, "image/png"),
        "prompt": _build_prompt(detections),
        "quality": config.quality,
        "size": output_size,
        "output_format": "png",
        "background": "opaque",
        "n": 1,
        "timeout": config.timeout,
    }
    # gpt-image-2는 모든 입력을 고충실도로 처리하므로 input_fidelity를 허용하지 않는다.
    if not config.model.startswith("gpt-image-2"):
        request["input_fidelity"] = "high"

    try:
        response = client.images.edit(**request)
    except Exception as exc:
        raise _provider_error(exc) from exc

    try:
        encoded = response.data[0].b64_json
        if not encoded:
            raise ValueError("empty image response")
        generated_bytes = base64.b64decode(encoded, validate=True)
        generated = Image.open(io.BytesIO(generated_bytes))
        generated.load()
        generated = generated.convert("RGB")
    except (AttributeError, IndexError, ValueError, binascii.Error, UnidentifiedImageError) as exc:
        raise RestorationError() from exc

    if generated.size != api_image.size:
        generated = generated.resize(api_image.size, Image.Resampling.LANCZOS)
    generated = generated.crop(content_box)
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    blend_mask = edit_mask
    if config.mask_blur_radius > 0:
        blend_mask = blend_mask.filter(ImageFilter.GaussianBlur(config.mask_blur_radius))
    return Image.composite(generated, original, blend_mask)
