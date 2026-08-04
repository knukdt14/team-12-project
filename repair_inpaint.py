"""
AI 복원 예상 이미지 생성 모듈
- YOLO가 검출한 bbox들을 기준으로 손상 부위 주변만 crop
- 사전학습된 Stable Diffusion Inpainting 모델로 손상 영역 복원
- 생성 결과를 원본 이미지의 해당 영역에 다시 합성

주의:
- 실제 정비 결과를 예측하는 모델이 아니라 "복원 시뮬레이션"입니다.
- 같은 차량의 형상/색상 보존을 위해 전체 이미지를 재생성하지 않고
  손상 bbox 주변 영역만 inpainting 합니다.

이 파일은 아래 문제들을 해결하기 위해 여러 차례 수정되었습니다:
1. bbox 1개만 쓰면 늘어진 파편(범퍼 하단 등)이 마스크 밖에 남는 문제
   -> generate_repaired_image_multi 로 여러 박스를 하나의 마스크로 합침
2. bbox 상하좌우를 동일 비율로 확장하면 위쪽(그릴/엠블럼)까지 침범하는 문제
   -> top/side 패딩과 "하단은 크롭 끝까지" 확장을 분리
3. 크롭을 정사각형으로 강제 리사이즈하면 그릴/범퍼가 왜곡되는 문제
   -> _resize_pair 를 비율 유지 방식으로 변경
4. 마스크가 우측 하단 워터마크까지 덮으면 SD가 깨진 글자를 생성하는 문제
   -> _make_crop_mask_safe 로 원본 이미지 절대 좌표 기준 워터마크 구역을
      마스크에서 강제 제외
5. 마스크가 헤드라이트/그릴 경계를 침범해 형태가 왜곡되는 문제
   -> negative_prompt에 왜곡 방지 문구 추가 + side_pad_ratio 축소
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-inpainting"


@dataclass
class RepairConfig:
    model_id: str = DEFAULT_MODEL_ID
    context_ratio: float = 0.55       # 크롭 범위. 늘어진 파편까지 포함하도록 넉넉히
    damage_pad_ratio: float = 0.10    # generate_repaired_image(단일 박스용) 에서만 사용
    mask_blur_radius: int = 9
    target_size: int = 512
    steps: int = 30
    guidance_scale: float = 9.0
    strength: float = 0.80
    seed: int = 42


PROMPT_SUFFIX = ", unchanged original badge and logo, unchanged grille pattern"

NEGATIVE_PROMPT_SUFFIX = (
    ", colorful logo, blue and yellow badge, novelty emblem, "
    "off-road bumper, extra vents, custom aftermarket bumper, "
    "changed badge color, wrong logo shape, "
    "distorted headlight, warped headlight, glowing orb, "
    "asymmetric headlight, melted headlight, blurry headlight lens"
)


def load_inpaint_pipeline(model_id: str = DEFAULT_MODEL_ID):
    """
    diffusers 모델 로드.
    CUDA 가능 시 float16 + GPU, 아니면 float32 + CPU.
    """
    try:
        from diffusers import AutoPipelineForInpainting
    except ImportError as exc:
        raise RuntimeError(
            "diffusers가 설치되어 있지 않습니다. "
            "`python -m pip install diffusers transformers accelerate safetensors` "
            "를 실행하세요."
        ) from exc

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    if use_cuda:
        pipe = pipe.to("cuda")
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    else:
        pipe = pipe.to("cpu")

    return pipe


def _clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _expand_box(box, width: int, height: int, ratio: float):
    x1, y1, x2, y2 = _clip_box(box, width, height)
    bw, bh = x2 - x1, y2 - y1

    return (
        max(0, int(x1 - bw * ratio)),
        max(0, int(y1 - bh * ratio)),
        min(width, int(x2 + bw * ratio)),
        min(height, int(y2 + bh * ratio)),
    )


def _make_crop_mask(
    crop_size,
    damage_box_in_crop,
    pad_ratio: float,
    blur_radius: int,
):
    cw, ch = crop_size
    x1, y1, x2, y2 = [float(v) for v in damage_box_in_crop]
    bw, bh = x2 - x1, y2 - y1

    x1 = max(0, int(x1 - bw * pad_ratio))
    y1 = max(0, int(y1 - bh * pad_ratio))
    x2 = min(cw, int(x2 + bw * pad_ratio))
    y2 = min(ch, int(y2 + bh * pad_ratio))

    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    return mask


def _make_crop_mask_safe(mask, crop_box, original_size, watermark_frac: float = 0.12):
    """원본 이미지 우측 하단 watermark_frac 영역을 마스크에서 강제로 제외.

    크롭 상대 비율로 마진을 주면 크롭 위치에 따라 워터마크가 여전히
    마스크에 걸릴 수 있어서, 항상 원본 이미지 절대 좌표 기준으로 계산합니다.
    """
    cx1, cy1, cx2, cy2 = crop_box
    ow, oh = original_size

    wm_x = ow * (1 - watermark_frac)
    wm_y = oh * (1 - watermark_frac)

    local_x = max(0, wm_x - cx1)
    local_y = max(0, wm_y - cy1)

    if local_x < mask.size[0] and local_y < mask.size[1]:
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [local_x, local_y, mask.size[0], mask.size[1]],
            fill=0,
        )

    return mask


def _resize_pair(image: Image.Image, mask: Image.Image, size: int):
    """비율을 유지한 채 리사이즈.

    이전에는 (size, size)로 강제 압축해서 크롭이 정사각형이 아닐 때
    그릴/범퍼 형태가 눌리거나 늘어나 보이는 왜곡이 있었습니다.
    diffusers inpaint 파이프라인은 정사각형이 아니어도 8의 배수 해상도면
    정상 동작하므로, 비율은 유지하고 스케일만 맞춥니다.
    """
    w, h = image.size
    scale = size / max(w, h)
    new_w = max(8, int(round(w * scale / 8) * 8))
    new_h = max(8, int(round(h * scale / 8) * 8))

    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_resized = mask.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image_resized, mask_resized


def _build_prompts(part_label: str, damage_label: str):
    prompt = (
        f"professional automotive collision repair, "
        f"repair the damaged {part_label} of the same exact vehicle, "
        f"remove all visible {damage_label}, "
        "restore the original factory body panel, "
        "intact front bumper, intact body panel, "
        "clean continuous automotive surface, "
        "correct original vehicle geometry, "
        "same exact vehicle, "
        "same manufacturer design, "
        "same body shape, "
        "same paint color, "
        "same headlights, "
        "same grille, "
        "same wheels, "
        "same reflections, "
        "same lighting, "
        "same camera viewpoint, "
        "same background, "
        "photorealistic automotive repair photograph"
    ) + PROMPT_SUFFIX

    negative_prompt = (
        "person, human, man, woman, child, "
        "hand, hands, arm, arms, finger, fingers, "
        "leg, foot, body, face, head, "
        "clothes, clothing, shirt, fabric, cloth, "
        "bag, umbrella, object, foreign object, "
        "different car, different vehicle, different model, "
        "changed headlights, changed grille, changed wheels, "
        "changed logo, changed background, "
        "extra parts, missing parts, duplicated parts, "
        "distorted vehicle, deformed geometry, malformed bumper, "
        "dent, scratch, crack, broken panel, damaged vehicle, "
        "cartoon, illustration, painting, blurry, low quality"
    ) + NEGATIVE_PROMPT_SUFFIX

    return prompt, negative_prompt


def _run_pipe(pipe, prompt, negative_prompt, model_image, model_mask, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(config.seed)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=model_image,
        mask_image=model_mask,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    )

    # 일부 diffusers pipeline은 strength를 지원하지 않을 수 있어 fallback
    try:
        generated = pipe(**kwargs, strength=config.strength).images[0]
    except TypeError:
        generated = pipe(**kwargs).images[0]

    return generated


def generate_repaired_image(
    pipe,
    original_image: Image.Image,
    damage_box,
    part_label: str = "vehicle body panel",
    damage_label: str = "damage",
    config: RepairConfig | None = None,
):
    """
    원본 PIL 이미지 + YOLO bbox(1개) -> 복원 시뮬레이션 PIL 이미지 반환.

    단일 박스만 쓰기 때문에, 박스 밖으로 늘어진 파편은 복원되지 않을 수 있습니다.
    가능하면 generate_repaired_image_multi 사용을 권장합니다.
    """
    config = config or RepairConfig()

    original = original_image.convert("RGB")
    width, height = original.size

    dx1, dy1, dx2, dy2 = _clip_box(damage_box, width, height)
    cx1, cy1, cx2, cy2 = _expand_box(
        (dx1, dy1, dx2, dy2),
        width,
        height,
        config.context_ratio,
    )

    crop = original.crop((cx1, cy1, cx2, cy2))

    damage_in_crop = (
        dx1 - cx1,
        dy1 - cy1,
        dx2 - cx1,
        dy2 - cy1,
    )

    mask = _make_crop_mask(
        crop.size,
        damage_in_crop,
        config.damage_pad_ratio,
        config.mask_blur_radius,
    )
    mask = _make_crop_mask_safe(mask, (cx1, cy1, cx2, cy2), (width, height))

    model_image, model_mask = _resize_pair(crop, mask, config.target_size)

    prompt, negative_prompt = _build_prompts(part_label, damage_label)

    generated = _run_pipe(pipe, prompt, negative_prompt, model_image, model_mask, config)
    generated = generated.resize(crop.size, Image.Resampling.LANCZOS)

    blended_crop = Image.composite(generated, crop, mask)

    repaired = original.copy()
    repaired.paste(blended_crop, (cx1, cy1))

    return repaired, mask, crop


def generate_repaired_image_multi(
    pipe,
    original_image: Image.Image,
    damage_boxes: list,
    part_label: str = "vehicle body panel",
    damage_label: str = "damage",
    config: RepairConfig | None = None,
    top_pad_ratio: float = 0.03,
    side_pad_ratio: float = 0.15,
    bottom_pad_ratio: float = 0.35,
    extend_to_bottom: bool = True,
    watermark_frac: float = 0.12,
):
    """
    원본 PIL 이미지 + YOLO bbox 여러 개 -> 복원 시뮬레이션 PIL 이미지 반환.

    - damage_boxes: 같은 손상 부위로 볼 [(x1,y1,x2,y2), ...] 전체를 넘기면
      하나의 마스크로 합쳐서 inpainting 합니다.
    - top_pad_ratio: 위쪽(그릴/엠블럼 방향) 패딩. 작게 유지해 로고 훼손을 방지.
    - side_pad_ratio: 좌우 패딩. 너무 크면 헤드라이트를 침범할 수 있음.
    - bottom_pad_ratio / extend_to_bottom: extend_to_bottom=True 이면 비율 대신
      크롭 하단까지 마스크를 밀어붙여, 늘어진 배선/파편처럼 박스보다
      훨씬 길게 튀어나온 손상도 확실히 덮습니다.
    - watermark_frac: 원본 이미지 우측 하단 이 비율만큼은 항상 마스크에서 제외
      (워터마크 위치에 깨진 글자가 생성되는 것을 방지).
    """
    config = config or RepairConfig()
    original = original_image.convert("RGB")
    width, height = original.size

    clipped = [_clip_box(b, width, height) for b in damage_boxes]

    ux1 = min(b[0] for b in clipped)
    uy1 = min(b[1] for b in clipped)
    ux2 = max(b[2] for b in clipped)
    uy2 = max(b[3] for b in clipped)

    cx1, cy1, cx2, cy2 = _expand_box(
        (ux1, uy1, ux2, uy2), width, height, config.context_ratio
    )
    crop = original.crop((cx1, cy1, cx2, cy2))
    cw, ch = crop.size

    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in clipped:
        bw, bh = x2 - x1, y2 - y1
        px1 = max(0, x1 - cx1 - bw * side_pad_ratio)
        py1 = max(0, y1 - cy1 - bh * top_pad_ratio)
        px2 = min(cw, x2 - cx1 + bw * side_pad_ratio)
        py2 = ch if extend_to_bottom else min(ch, y2 - cy1 + bh * bottom_pad_ratio)
        draw.rectangle([px1, py1, px2, py2], fill=255)

    # 워터마크 구역은 항상 제외 (블러 넣기 전에 적용)
    mask = _make_crop_mask_safe(
        mask, (cx1, cy1, cx2, cy2), (width, height), watermark_frac
    )

    if config.mask_blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(config.mask_blur_radius))

    model_image, model_mask = _resize_pair(crop, mask, config.target_size)

    prompt, negative_prompt = _build_prompts(part_label, damage_label)

    generated = _run_pipe(pipe, prompt, negative_prompt, model_image, model_mask, config)
    generated = generated.resize(crop.size, Image.Resampling.LANCZOS)

    blended_crop = Image.composite(generated, crop, mask)

    repaired = original.copy()
    repaired.paste(blended_crop, (cx1, cy1))

    return repaired, mask, crop
