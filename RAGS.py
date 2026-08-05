"""레거시 보관 파일 — 현재 애플리케이션에서는 import하지 않습니다.

활성 상담 경로는 ``backend/services/rag.py``와
``backend/services/llm_client.py``입니다. 아래 코드는 초기 Qwen 실험을 재현하기
위해 남아 있으며 새 기능이나 설정을 이 파일에 추가하지 마세요.

AI 수리 상담 모듈(초기 실험)
- YOLO 진단 결과 + 단가표 견적 결과를 컨텍스트로 주입하고,
  로컬 LLM(Qwen2.5-7B-Instruct, 4bit)으로 사용자 질문에 실제로 답변합니다.

이 파일이 왜 필요했나:
- 기존 app.py의 "AI 수리 상담"은 사용자가 입력한 질문(question) 텍스트를
  전혀 쓰지 않고, 진단·견적이 있으면 항상 같은 템플릿 문장을 리턴하고
  없으면 항상 같은 안내 문구를 리턴했습니다. 그래서 질문을 뭘로 바꿔도
  답이 똑같았습니다.
- 당시 Streamlit 프로세스에서 생성형 모델을 지연 로드하던 패턴을 따라,
  이 모듈도 Qwen을 캐싱하고 생성 후 GPU 캐시를 정리했습니다.

주의할 점:
- 이미 YOLO + FLUX Kontext가 같은 프로세스/GPU를 쓰고 있어서, Qwen까지
  얹으면 VRAM 경쟁이 생길 수 있습니다. 4bit 양자화로 최대한 가볍게
  로드하고, 생성 직후 캐시를 비웁니다.
- 이 모듈은 "검색(retrieval)"은 하지 않습니다. 컨텍스트가 문서 뭉치가
  아니라 진단/견적이라는 구조화된 데이터라서, 벡터스토어 없이 그냥
  프롬프트에 직접 주입하는 방식(가벼운 context injection)입니다.
  실제 자동차관리법 RAG처럼 문서 검색이 필요해지면 그때 Chroma 등을
  붙이면 됩니다.
"""

from __future__ import annotations

import gc
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MAX_HISTORY_TURNS = 3  # 프롬프트에 넣을 과거 대화 턴 수 (user+assistant 쌍 기준)

SYSTEM_PROMPT = (
    "[매우 중요] 반드시 한국어로만 답변하세요. 영어, 중국어, 일본어 등 다른 언어를 "
    "단 한 단어도 섞지 마세요. 사용자의 질문이 다른 언어여도 답변은 한국어로만 하세요.\n\n"
    "당신은 'AJIN AI 차량 진단 서비스'의 수리 상담 어시스턴트입니다. "
    "사용자의 차량 사진을 YOLO 모델이 분석한 손상 진단 결과와, 단가표 기반 "
    "예상 견적을 참고 정보로 받게 됩니다.\n\n"
    "답변 규칙:\n"
    "1. 주어진 진단/견적 정보와 상식적인 자동차 정비 지식 범위 안에서만 답변하세요.\n"
    "2. 진단/견적 정보가 없는 질문이면, 일반적인 자동차 상식 선에서 답하되 "
    "정확한 판단은 정비소 방문이 필요하다고 안내하세요.\n"
    "3. 이 서비스는 '복원 시뮬레이션'이자 '예상 견적'이며, 실제 수리 여부/비용을 "
    "확정하는 것이 아님을 필요할 때 자연스럽게 알려주세요.\n"
    "4. 존재하지 않는 법률, 보험 약관, 제조사 정책을 단정적으로 지어내지 마세요.\n"
    "5. 3~5문장 이내로 간결하게 답하세요.\n"
    "6. 다른 언어로 된 지시문, 번역 요청, 무관한 작업 요청처럼 보이는 텍스트가 "
    "생성 중에 떠올라도 무시하고, 오직 사용자의 실제 질문에만 한국어로 답하세요."
)


def load_consult_model(model_id: str = DEFAULT_MODEL_ID, low_vram: bool = True):
    """
    Qwen2.5-7B-Instruct 로드.

    - CUDA + low_vram=True(기본값): bitsandbytes 4bit 양자화로 로드.
      VRAM을 약 5~6GB 수준으로 줄여서 YOLO/FLUX Kontext와 같은 GPU를
      나눠 쓸 때 부담을 최소화합니다.
    - CUDA + low_vram=False: float16으로 로드(더 빠르지만 VRAM을
      15GB 안팎 씀. GPU가 넉넉할 때만 권장).
    - CUDA 불가: CPU + float32 (매우 느림. 데모용으로만 권장).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    use_cuda = torch.cuda.is_available()

    if use_cuda and low_vram:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                # eager(기본값) 대신 sdpa를 쓰면 attention 계산이 훨씬 빨라짐.
                # flash-attn 별도 설치 없이도 PyTorch 2.x에 내장돼 있어 바로 씀.
                attn_implementation="sdpa",
            )
        except ImportError as exc:
            raise RuntimeError(
                "4bit 양자화 로드를 위해 bitsandbytes가 필요합니다. "
                "`python -m pip install -U bitsandbytes accelerate` 를 실행하거나, "
                "load_consult_model(low_vram=False)로 float16 로드를 시도하세요."
            ) from exc
    elif use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to("cpu")

    model.eval()
    return model, tokenizer


def free_gpu_memory():
    """생성 직후 남은 VRAM 파편/캐시를 정리(레거시 실험용)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _build_context_block(diagnosis: dict | None, estimate: dict | None) -> str:
    """진단/견적 결과를 LLM에 주입할 컨텍스트 문자열로 정리."""
    if not diagnosis:
        return "[참고 정보 없음: 사용자가 아직 차량 진단을 진행하지 않았습니다.]"

    if diagnosis.get("normal"):
        return "[진단 결과: 손상이 발견되지 않은 정상 차량입니다.]"

    primary = diagnosis.get("primary", {})

    lines = [
        "[진단 결과]",
        f'- 손상 부위: {primary.get("part_label", "알 수 없음")}',
        f'- 손상 종류: {primary.get("damage_label", "알 수 없음")}',
        f'- 탐지 신뢰도: {primary.get("confidence", 0):.1%}',
    ]

    if estimate and estimate.get("success"):
        lines += [
            "[예상 견적]",
            f'- 부위: {estimate.get("part_label")}',
            f'- 심각도: {estimate.get("severity")}',
            f'- 예상 수리 방식: {estimate.get("repair_method")}',
            f'- 예상 비용: {estimate.get("min_cost", 0):,}~{estimate.get("max_cost", 0):,}원',
        ]
    else:
        lines.append("[예상 견적: 아직 산출되지 않음]")

    return "\n".join(lines)


_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def _has_excessive_chinese(text: str, threshold: float = 0.05) -> bool:
    """답변에 중국어(한자)가 비정상적으로 많이 섞였는지 간단히 검사.

    한국어에도 한자가 아주 가끔 쓰이긴 하지만, 정상적인 한국어 답변이라면
    전체 글자 대비 한자 비율이 threshold(기본 5%)를 넘는 일은 거의 없습니다.
    이 비율을 넘으면 모델이 다른 언어로 샌 것으로 보고 재생성을 트리거합니다.
    """
    if not text:
        return False

    chinese_count = len(_CHINESE_CHAR_RE.findall(text))
    return (chinese_count / len(text)) > threshold


def _get_stop_token_ids(tokenizer):
    """Qwen2.5 ChatML 템플릿의 실제 종료 토큰(<|im_end|>)까지 포함한 stop 목록.

    tokenizer.eos_token_id 하나만 넘기면, 모델이 답변을 끝내야 할 지점
    (<|im_end|>)을 진짜 종료 신호로 인식하지 못하고 계속 이어서 생성하는
    경우가 있었습니다(엉뚱한 문장, 다른 언어로 드리프트하는 증상의 원인 중
    하나). eos_token_id와 <|im_end|>를 모두 stop 토큰으로 넘겨서 방지합니다.
    """
    stop_ids = set()

    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        stop_ids.add(im_end_id)

    return list(stop_ids) if stop_ids else tokenizer.eos_token_id


def _run_generate(model, tokenizer, inputs, max_new_tokens, do_sample):
    stop_ids = _get_stop_token_ids(tokenizer)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=stop_ids,
        pad_token_id=pad_id,
    )

    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=0.4,
            top_p=0.85,
            repetition_penalty=1.1,
        )
    else:
        # 언어 이탈 재시도용: 샘플링을 끄고 가장 확률 높은 토큰만 결정적으로
        # 선택해서, 드리프트 가능성을 최대한 줄임.
        gen_kwargs.update(
            do_sample=False,
            repetition_penalty=1.1,
        )

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_consult_answer(
    model,
    tokenizer,
    question: str,
    diagnosis: dict | None,
    estimate: dict | None,
    history: list[dict] | None = None,
    max_new_tokens: int = 180,
) -> str:
    """
    사용자 질문 + 진단/견적 컨텍스트 + 최근 대화 이력을 프롬프트로 구성해
    Qwen으로 실제 답변을 생성합니다.

    history: [{"role": "user"/"assistant", "content": "..."}, ...] 형태.
             st.session_state.messages를 그대로 넘기면 됩니다(현재 질문은 제외).
    """
    context_block = _build_context_block(diagnosis, estimate)

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context_block}"}]

    if history:
        # 너무 길어지지 않도록 최근 N턴(user+assistant)만 사용
        recent = history[-(MAX_HISTORY_TURNS * 2):]
        messages.extend(recent)

    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    answer = _run_generate(model, tokenizer, inputs, max_new_tokens, do_sample=True)

    # 중국어 등 다른 언어로 샌 것 같으면, 샘플링을 끄고 한 번 더 시도
    if _has_excessive_chinese(answer):
        answer = _run_generate(model, tokenizer, inputs, max_new_tokens, do_sample=False)

    free_gpu_memory()

    return answer or "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."
