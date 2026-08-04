"""
AI 복원 예상 이미지 생성 모듈
- YOLO가 검출한 bbox들을 기준으로 손상 부위 주변만 crop
- FLUX.1 Kontext (black-forest-labs/FLUX.1-Kontext-dev)로 손상 영역을
  자연어 지시문(prompt) 기반으로 복원
- 생성 결과를 원본 이미지의 해당 영역에 다시 합성

주의:
- 실제 정비 결과를 예측하는 모델이 아니라 "복원 시뮬레이션"입니다.
- 같은 차량의 형상/색상 보존을 위해 전체 이미지를 재생성하지 않고
  손상 bbox 주변 영역만 편집한 뒤 원본에 다시 붙여넣습니다.

이 파일은 아래 문제들을 해결하기 위해 여러 차례 수정되었습니다:
1. bbox 1개만 쓰면 늘어진 파편(범퍼 하단 등)이 마스크 밖에 남는 문제
   -> generate_repaired_image_multi 로 여러 박스를 하나의 마스크로 합침
2. bbox 상하좌우를 동일 비율로 확장하면 위쪽(그릴/엠블럼)까지 침범하는 문제
   -> top/side 패딩과 "하단은 크롭 끝까지" 확장을 분리
3. 크롭을 정사각형으로 강제 리사이즈하면 그릴/범퍼가 왜곡되는 문제
   -> 비율 유지 방식으로 리사이즈
4. 마스크가 우측 하단 워터마크까지 덮으면 생성 모델이 깨진 글자를 만드는 문제
   -> _make_crop_mask_safe 로 원본 이미지 절대 좌표 기준 워터마크 구역을
      마스크에서 강제 제외
5. 마스크가 헤드라이트/그릴 경계를 침범해 형태가 왜곡되는 문제
   -> negative_prompt에 왜곡 방지 문구 추가 + side_pad_ratio 축소
6. Stable Diffusion Inpainting -> FLUX.1 Kontext 로 교체
   -> Kontext는 mask_image를 받는 인페인팅 파이프라인이 아니라, 이미지 전체 +
      자연어 편집 지시문을 입력받아 "지시된 부분만 바꾸고 나머지는 그대로
      유지"하도록 학습된 in-context 편집 모델입니다. 그래서:
        * pipe 호출 시 mask_image/strength 파라미터가 없어졌습니다.
        * 대신 자체적으로 만드는 마스크(_make_crop_mask 계열)는 "모델에게 알려주는
          용도"가 아니라, 생성 결과와 원본 크롭을 합성(Image.composite)할 때
          손상 영역 바깥은 원본 픽셀을 100% 그대로 쓰기 위한 안전장치로만 씁니다.
        * negative_prompt는 Kontext가 guidance-distilled 모델이라 기본적으로
          무시되며, true_cfg_scale(RepairConfig.true_cfg_scale) > 1.0 으로 설정
          해야 실제로 반영됩니다(대신 추론 속도가 약 2배 느려집니다).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"


@dataclass
class RepairConfig:
    model_id: str = DEFAULT_MODEL_ID
    context_ratio: float = 0.55       # 크롭 범위. 늘어진 파편까지 포함하도록 넉넉히
    damage_pad_ratio: float = 0.10    # generate_repaired_image(단일 박스용) 에서만 사용
    mask_blur_radius: int = 9
    target_size: int = 1024           # FLUX Kontext는 1024 부근에서 최적. 512도 동작은 함
    steps: int = 28                   # Kontext 공식 예제 권장값
    guidance_scale: float = 2.5       # Kontext 권장값 (SD의 7~9 스케일과는 다름)
    true_cfg_scale: float = 1.0       # 1.0이면 negative_prompt 비활성화(기본, 더 빠름)
    low_vram: bool = True             # True면 enable_model_cpu_offload 사용 (12B 모델 VRAM 절약)
    seed: int = 42


PROMPT_SUFFIX = ", unchanged original badge and logo, unchanged grille pattern"

NEGATIVE_PROMPT_SUFFIX = (
    ", colorful logo, blue and yellow badge, novelty emblem, "
    "off-road bumper, extra vents, custom aftermarket bumper, "
    "changed badge color, wrong logo shape, "
    "distorted headlight, warped headlight, glowing orb, "
    "asymmetric headlight, melted headlight, blurry headlight lens"
)


def load_inpaint_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    low_vram: bool = True,
    sequential_offload: bool = True,
):
    """
    FLUX.1 Kontext diffusers 파이프라인 로드.

    - CUDA 가능 시 bfloat16, 아니면 float32 + CPU (CPU는 매우 느립니다).
    - low_vram=True인 경우, 세 단계 중 하나를 씁니다 (VRAM이 적을수록 아래로):
        1) sequential_offload=True (기본값): enable_sequential_cpu_offload().
           레이어 단위로 필요할 때만 GPU에 올리고 바로 내려서 순간 최대
           VRAM 사용량이 가장 작습니다(수 GB 수준). 대신 가장 느립니다.
           8GB급 GPU에서는 이 모드를 권장합니다.
        2) sequential_offload=False: enable_model_cpu_offload().
           텍스트 인코더/트랜스포머/VAE 같은 큰 서브모듈 단위로 올렸다
           내려서 sequential보다 빠르지만, 순간 최대 VRAM 사용량이 더
           큽니다(가장 큰 서브모듈 하나가 통째로 올라감). 16GB 이상
           GPU에 권장합니다.
    - low_vram=False: pipe.to("cuda")로 전부 GPU에 상주. 24GB 이상 GPU에서
      가장 빠르지만, 다른 모델(YOLO 등)과 VRAM을 나눠 써야 하면 위험합니다.
    - VAE에 slicing/tiling을 켜서 디코드 단계의 순간 메모리도 줄입니다.
    - FLUX.1-Kontext-dev는 gated model이라 최초 1회
      `hf auth login` 등으로 라이선스 동의 + 토큰 인증이 필요합니다.
    """
    try:
        from diffusers import FluxKontextPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers에서 FluxKontextPipeline을 불러올 수 없습니다. "
            "FLUX.1 Kontext는 최신 diffusers가 필요할 수 있습니다. 아래를 실행하세요:\n"
            "python -m pip install -U git+https://github.com/huggingface/diffusers.git "
            "transformers accelerate safetensors sentencepiece protobuf\n"
            "그리고 https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev 에서 "
            "라이선스에 동의한 뒤 `hf auth login`으로 인증하세요."
        ) from exc

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    if use_cuda:
        if low_vram and sequential_offload:
            pipe.enable_sequential_cpu_offload()
        elif low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass

    return pipe


def free_gpu_memory():
    """생성 직후 남은 VRAM 파편/캐시를 정리.

    st.cache_resource로 파이프라인을 캐싱해두면 프로세스가 살아있는 동안
    계속 상주하기 때문에, 매 생성 후 명시적으로 비워주지 않으면 다음번
    YOLO 탐지 등 다른 GPU 작업이 실패할 수 있습니다.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
    """손상 영역 마스크 생성.

    주의: 이 마스크는 모델에 입력되지 않습니다(Kontext는 mask_image를
    받지 않음). 생성 결과와 원본 크롭을 합성할 때, 이 마스크 안쪽만
    생성 결과를 쓰고 바깥쪽은 원본 픽셀을 그대로 유지하기 위한
    블렌딩 전용 마스크입니다.
    """
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


def _resize_for_model(image: Image.Image, size: int):
    """비율을 유지한 채 모델 입력 해상도로 리사이즈.

    (size, size)로 강제 압축하면 크롭이 정사각형이 아닐 때 그릴/범퍼
    형태가 눌리거나 늘어나 보이는 왜곡이 있어서 비율은 유지하고
    스케일만 맞춥니다. FLUX 계열은 VAE 다운샘플(8배) + 2x2 patchify 때문에
    가로/세로가 16의 배수여야 안전하게 동작합니다(SD의 8배수보다 더 큰 배수).
    """
    w, h = image.size
    scale = size / max(w, h)
    new_w = max(16, int(round(w * scale / 16) * 16))
    new_h = max(16, int(round(h * scale / 16) * 16))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _build_prompts(part_label: str, damage_label: str):
    """FLUX Kontext는 instruction 스타일 편집 프롬프트에 잘 반응합니다.

    (SD Inpainting 시절의 키워드 나열형 프롬프트 대신, "무엇을 어떻게 바꿔라"를
    명확한 문장으로 지시하고, 나머지는 그대로 유지하라고 명시합니다.)
    """
    prompt = (
        f"Repair the damaged {part_label} in this photo. "
        f"Completely remove the {damage_label} and restore that area to a clean, "
        "intact factory body panel, as if it just left a professional collision "
        "repair shop. Keep everything else in the image exactly the same: "
        "the same exact vehicle, same manufacturer design, same body shape, "
        "same paint color, same headlights, same grille, same wheels, "
        "same reflections, same lighting, same camera viewpoint, same background. "
        "Only change the damaged area, nothing else."
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


def _run_pipe(pipe, prompt, negative_prompt, model_image, config):
    """파이프라인 실행.

    중요: width/height를 명시하지 않으면 FluxKontextPipeline이 입력을
    자체 선호 해상도(PREFERRED_KONTEXT_RESOLUTIONS)로 다시 리사이즈해서
    출력 종횡비가 입력과 달라질 수 있습니다. 그 결과를 원본 크기로
    되돌려 붙이면 차량이 확대/이동된 것처럼 어긋나 보이는(정합 깨짐)
    문제가 생기므로, 반드시 입력과 동일한 크기를 명시합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(config.seed)

    mw, mh = model_image.size

    kwargs = dict(
        image=model_image,
        prompt=prompt,
        width=mw,
        height=mh,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.steps,
        generator=generator,
    )

    # FLUX Kontext는 guidance-distilled 모델이라 negative_prompt는
    # true_cfg_scale > 1.0일 때만 실제로 반영됩니다(기본 1.0이면 무시).
    # true_cfg_scale > 1이면 스텝당 2번(조건/무조건) 추론하므로 약 2배 느려집니다.
    if config.true_cfg_scale > 1.0:
        kwargs["negative_prompt"] = negative_prompt
        kwargs["true_cfg_scale"] = config.true_cfg_scale

    return pipe(**kwargs).images[0]


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

    model_image = _resize_for_model(crop, config.target_size)

    prompt, negative_prompt = _build_prompts(part_label, damage_label)

    generated = _run_pipe(pipe, prompt, negative_prompt, model_image, config)
    free_gpu_memory()
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
      하나의 블렌딩 마스크로 합쳐서 편집합니다.
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

    model_image = _resize_for_model(crop, config.target_size)

    prompt, negative_prompt = _build_prompts(part_label, damage_label)

    generated = _run_pipe(pipe, prompt, negative_prompt, model_image, config)
    free_gpu_memory()
    generated = generated.resize(crop.size, Image.Resampling.LANCZOS)

    blended_crop = Image.composite(generated, crop, mask)

    repaired = original.copy()
    repaired.paste(blended_crop, (cx1, cy1))

    return repaired, mask, crop


def _build_prompts_full(damaged_parts: str):
    """전체 이미지 편집용 프롬프트.

    크롭 방식과 달리 차량 전체 맥락이 보이므로, "손상된 앞부분 전체를
    한 대의 온전한 차로 복원하라"고 지시합니다. 검출된 부위 목록
    (damaged_parts)을 넣어 무엇이 손상 대상인지 명확히 알려줍니다.
    """
    prompt = (
        f"This photo shows a crashed car with severe collision damage "
        f"({damaged_parts}). "
        "Fully repair the car: restore the crumpled hood, broken front bumper, "
        "grille, headlights, fenders and any hanging or missing parts to a "
        "clean, undamaged factory condition, as if professionally repaired. "
        "The repaired car must be the exact same vehicle: same manufacturer, "
        "same model, same body shape and proportions, same paint color, "
        "same license plate, in the exact same position and pose. "
        "Keep the camera viewpoint, lighting, ground, background, and all "
        "other cars in the photo exactly the same. "
        "Do not move, rotate, or resize the car."
    ) + PROMPT_SUFFIX

    negative_prompt = (
        "different car, different model, moved car, rotated car, zoomed car, "
        "changed background, changed other cars, "
        "dent, scratch, crack, broken panel, hanging parts, debris, "
        "damaged vehicle, deformed geometry, "
        "cartoon, illustration, painting, blurry, low quality"
    ) + NEGATIVE_PROMPT_SUFFIX

    return prompt, negative_prompt


def generate_repaired_image_full(
    pipe,
    original_image: Image.Image,
    damage_boxes: list,
    damaged_parts: str = "front end damage",
    config: RepairConfig | None = None,
    side_pad_ratio: float = 0.45,
    top_pad_ratio: float = 0.20,
    bottom_pad_ratio: float = 1.10,
    merge_boxes: bool = True,
    watermark_frac: float = 0.12,
):
    """
    원본 전체 이미지를 그대로 FLUX Kontext에 넣어 복원하는 방식.

    기존 크롭 방식(generate_repaired_image_multi)의 문제:
    1. 크롭이 좁으면 모델이 차량 전체 형태를 못 봐서, 마스크 안에
       "다른 원근/크기의 차 앞부분"을 그려 넣어 원본과 어긋남.
    2. YOLO 박스가 손상 일부(예: 보닛)만 잡으면, 박스 밖의 부서진
       범퍼/헤드라이트가 마스크 밖에 남아 반쪽짜리 복원이 됨.

    이 함수의 해결책:
    - 이미지 전체를 (비율 유지, 16배수) 리사이즈해서 입력하고
      width/height를 명시 → 출력이 입력과 픽셀 단위로 정렬됨.
    - 마스크는 검출 박스들의 합집합을 상하좌우로 넉넉히 확장
      (기본: 좌우 45%, 위 20%, 아래 110%)해서 늘어진 파편·부서진
      범퍼까지 편집 허용 영역에 포함.
    - 합성은 원본 좌표계에서 수행하므로 크롭-붙여넣기 이음새가 없음.

    트레이드오프: 원본이 매우 크면 복원 영역 해상도가 target_size
    기준으로 제한되어 주변보다 약간 소프트해질 수 있으나, 정합이
    맞기 때문에 크롭 방식의 "확대된 다른 차" 현상보다 훨씬 자연스럽습니다.

    반환: (repaired, mask, generated_full)
    """
    config = config or RepairConfig()
    original = original_image.convert("RGB")
    width, height = original.size

    clipped = [_clip_box(b, width, height) for b in damage_boxes]

    if merge_boxes and len(clipped) > 1:
        boxes = [(
            min(b[0] for b in clipped),
            min(b[1] for b in clipped),
            max(b[2] for b in clipped),
            max(b[3] for b in clipped),
        )]
    else:
        boxes = clipped

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        draw.rectangle(
            [
                max(0, x1 - bw * side_pad_ratio),
                max(0, y1 - bh * top_pad_ratio),
                min(width, x2 + bw * side_pad_ratio),
                min(height, y2 + bh * bottom_pad_ratio),
            ],
            fill=255,
        )

    # 워터마크 구역(우측 하단)은 항상 원본 유지
    if watermark_frac > 0:
        draw.rectangle(
            [
                width * (1 - watermark_frac),
                height * (1 - watermark_frac),
                width,
                height,
            ],
            fill=0,
        )

    # 이미지 크기에 비례한 페더링으로 경계를 부드럽게
    blur = max(config.mask_blur_radius, int(min(width, height) * 0.02))
    mask = mask.filter(ImageFilter.GaussianBlur(blur))

    model_image = _resize_for_model(original, config.target_size)

    prompt, negative_prompt = _build_prompts_full(damaged_parts)

    generated = _run_pipe(pipe, prompt, negative_prompt, model_image, config)
    free_gpu_memory()

    # 입력과 동일 종횡비/크기로 생성됐으므로 원본 크기로 되돌리면 정렬됨
    generated = generated.resize((width, height), Image.Resampling.LANCZOS)

    repaired = Image.composite(generated, original, mask)

    return repaired, mask, generated
