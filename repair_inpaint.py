"""
AI 복원 예상 이미지 생성 모듈
- YOLO가 검출한 bbox들을 기준으로 손상 부위 마스크 생성
- SD1.5 Inpainting (stable-diffusion-v1-5/stable-diffusion-inpainting)으로
  마스크 영역만 복원 생성
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
6. Stable Diffusion Inpainting -> FLUX.1 Kontext 로 교체했다가, 다시 SD1.5
   Inpainting 으로 되돌림 (팀 표준 모델에 맞춤)
   -> FLUX.1 Kontext는 품질은 좋지만 gated repo(라이선스 동의 필요) + 약 24GB +
      GPU 사실상 필수라서, 팀원 간 재현과 클라우드 배포가 어려웠습니다.
   -> SD1.5 Inpainting은 약 2GB에 공개 모델이라 팀원 누구나 같은 결과를 얻을 수
      있습니다. 대신 mask_image를 반드시 넘겨야 하는 "진짜" 인페인팅 모델이라,
      마스크의 역할이 아래처럼 이원화됩니다:
        * pipe에 넘기는 마스크 = "여기를 새로 그려라"라는 지시 (흰색=생성 대상)
        * 합성(Image.composite)에 쓰는 마스크 = 생성 결과 중 어디를 채택할지.
          마스크 바깥은 원본 픽셀을 100% 유지하는 안전장치.
      두 용도에 같은 마스크를 쓰되, pipe 입력용은 블러를 빼고(경계가 흐리면
      모델이 어디까지 그릴지 헷갈려 함) 합성용만 페더링합니다.
        * negative_prompt가 정상 동작합니다(SD1.5는 classifier-free guidance를
          그대로 쓰므로 true_cfg_scale 같은 우회가 필요 없습니다).
        * strength로 원본을 얼마나 남길지 조절합니다(1.0=완전 새로 생성).
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


# SD1.5 Inpainting.
#
# 원본 `runwayml/stable-diffusion-inpainting` 저장소는 내려가서, 아래 공식
# 커뮤니티 재업로드본을 기본값으로 씁니다. 팀에서 다른 미러를 쓰고 있다면
# 환경변수로 덮어쓰세요:
#     REPAIR_MODEL_ID=benjamin-paine/stable-diffusion-v1-5-inpainting
DEFAULT_MODEL_ID = os.getenv(
    "REPAIR_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-inpainting"
)


@dataclass
class RepairConfig:
    model_id: str = DEFAULT_MODEL_ID
    context_ratio: float = 0.55       # 크롭 범위. 늘어진 파편까지 포함하도록 넉넉히
    damage_pad_ratio: float = 0.10    # generate_repaired_image(단일 박스용) 에서만 사용
    mask_blur_radius: int = 9
    # SD1.5는 512x512로 학습됐습니다. 1024를 넣으면 같은 물체가 두 번 그려지는
    # (duplication) 현상이 잘 나므로 512를 기본으로 둡니다.
    target_size: int = 512
    steps: int = 35                   # SD1.5 인페인팅 권장 30~50
    guidance_scale: float = 7.5       # SD1.5 표준 CFG (Kontext의 2.5와 다름)
    # 마스크 영역을 얼마나 새로 그릴지. 1.0이면 원본 픽셀을 무시하고 완전 재생성.
    # 손상 복원은 "원래 형태를 되살리는" 작업이라 0.9 정도가 안정적입니다.
    strength: float = 0.9
    low_vram: bool = True             # True면 attention slicing + cpu offload
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
    SD1.5 Inpainting diffusers 파이프라인 로드.

    - CUDA 가능 시 float16, 아니면 float32 + CPU.
      SD1.5는 약 2GB(fp16 기준 ~1GB)라 FLUX(24GB)와 달리 8GB급 GPU에 통째로 올라갑니다.
      CPU로도 512px / 35스텝이면 수 분 수준이라 GPU 없이도 시연은 가능합니다.
    - low_vram=True: attention slicing + VAE slicing으로 순간 메모리를 낮춥니다.
      YOLO/ResNet과 VRAM을 나눠 쓰는 상황이라 기본값으로 켜둡니다.
    - sequential_offload=True: 추가로 enable_sequential_cpu_offload()까지 씁니다.
      SD1.5에서는 보통 불필요하며(오히려 느려짐), 4GB 미만 GPU에서만 의미가 있습니다.
    - safety_checker=None: 차량 사진에 오탐이 걸려 검은 이미지가 반환되는 것을 막습니다.
      (수업용 데모라 무방합니다)
    - FLUX와 달리 gated repo가 아니라서 라이선스 동의나 토큰이 필요 없습니다.
    """
    try:
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers에서 StableDiffusionInpaintPipeline을 불러올 수 없습니다.\n"
            "python -m pip install -U diffusers transformers accelerate safetensors"
        ) from exc

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if use_cuda:
        if low_vram and sequential_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    if low_vram:
        try:
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()
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

    SD1.5 Inpainting에서 이 마스크는 두 곳에 쓰입니다.
    1. 모델 입력 — 흰색(255) 영역만 새로 그립니다. _run_pipe가 128 기준으로
       이진화해서 넘기므로, 여기서 넣은 블러는 모델 입력에는 영향이 없습니다.
    2. 합성 — 블러된 원본 그대로 Image.composite에 써서 경계를 부드럽게 잇습니다.
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
    """마스크 영역에 "무엇을 그려 넣을지"를 서술하는 프롬프트.

    SD1.5 Inpainting은 편집 지시문("~를 제거하라")이 아니라 결과물 묘사
    ("깨끗한 순정 패널")에 반응합니다. 마스크 바깥은 어차피 건드리지 않으므로
    "나머지는 그대로 두라"는 문장은 실질적인 효과가 없지만, 차량 종류·색상을
    유지시키는 힌트로는 작동해서 남겨둡니다.
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


def _run_pipe(pipe, prompt, negative_prompt, model_image, config, mask_image=None):
    """SD1.5 인페인팅 파이프라인 실행.

    mask_image는 필수입니다. SD 인페인팅은 "마스크가 흰색(255)인 픽셀만 새로
    그리고 검은색(0)은 원본을 유지"하는 모델이라, 마스크를 안 넘기면 동작
    자체가 성립하지 않습니다.

    마스크 전처리 두 가지:
    1. 모드를 "L"로 맞추고 model_image와 크기를 정확히 일치시킵니다.
       크기가 다르면 diffusers 내부에서 리사이즈되며 경계가 어긋납니다.
    2. 여기 들어오는 마스크는 블러를 빼고 이진에 가깝게 씁니다.
       경계가 흐리면 모델이 "어디까지 그려야 하는지"를 애매하게 해석해
       손상 흔적이 반쯤 남습니다. 부드러운 이음새는 합성 단계에서
       블러된 마스크로 따로 처리합니다.

    width/height를 명시하는 이유: 지정하지 않으면 파이프라인이 512로
    강제 리사이즈해서 출력 종횡비가 입력과 달라지고, 원본에 되붙일 때
    차량이 늘어나 보입니다.
    """
    if mask_image is None:
        raise ValueError(
            "SD1.5 인페인팅에는 mask_image가 필요합니다. "
            "호출부에서 손상 영역 마스크를 전달하세요."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(config.seed)

    mw, mh = model_image.size

    mask = mask_image.convert("L")
    if mask.size != (mw, mh):
        mask = mask.resize((mw, mh), Image.Resampling.NEAREST)
    # 블러 잔재를 제거해 경계를 또렷하게 (128 초과분만 생성 대상)
    mask = mask.point(lambda v: 255 if v > 128 else 0)

    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=model_image,
        mask_image=mask,
        width=mw,
        height=mh,
        guidance_scale=config.guidance_scale,
        strength=config.strength,
        num_inference_steps=config.steps,
        generator=generator,
    ).images[0]


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

    generated = _run_pipe(
        pipe, prompt, negative_prompt, model_image, config, mask_image=mask
    )
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

    generated = _run_pipe(
        pipe, prompt, negative_prompt, model_image, config, mask_image=mask
    )
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

    generated = _run_pipe(
        pipe, prompt, negative_prompt, model_image, config, mask_image=mask
    )
    free_gpu_memory()

    # 입력과 동일 종횡비/크기로 생성됐으므로 원본 크기로 되돌리면 정렬됨
    generated = generated.resize((width, height), Image.Resampling.LANCZOS)

    repaired = Image.composite(generated, original, mask)

    return repaired, mask, generated
