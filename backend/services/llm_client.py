"""Ollama REST 호출 클라이언트 (EXAONE-3.5-2.4B-Instruct, GGUF Q4).

Docker Compose 환경에서는 컨테이너 내부 네트워크로 http://llm:11434 를 쓰고,
로컬 단독 실행 시에는 http://localhost:11434 를 쓴다 — 환경변수 OLLAMA_BASE_URL로
오버라이드 가능(docker-compose.yml의 backend 서비스에서 주입).

장애 처리 방침:
    generate()는 예외를 던지지 않고 None을 반환한다. LLM이 아직 모델을
    내려받는 중(첫 기동 후 2~3분)이거나 죽어 있어도 상담 화면이 500/503으로
    깨지면 안 되기 때문이다. 호출부(routers/chat.py)가 검색 결과 기반
    폴백 답변으로 대체한다.

모델 선택:
    기본값은 exaone3.5:2.4b (LG AI연구원, 한국어 특화).
    qwen2.5 계열은 답변이 중국어로 새는 문제가 잦아 교체했다.
    사양이 부족하면 .env에서 OLLAMA_MODEL=qwen2.5:0.5b 로 낮출 수 있다.
"""
import os
from typing import List, Optional

import requests

# .strip()은 Windows 편집기가 .env에 남긴 CR(\r)이나 앞뒤 공백을 제거하기 위한 것.
# 이게 섞이면 "exaone3.5:2.4b\r" 같은 이름으로 요청해 404가 난다.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b").strip()

# CPU 추론이라 넉넉하게. 그래도 사용자가 무한정 기다리지 않도록 상한을 둔다.
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "400"))
# 기본 컨텍스트(2048)로는 참고 자료 3개 + 진단 결과가 잘려나간다.
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

print(f"[llm] 설정 — URL={OLLAMA_BASE_URL} MODEL={MODEL_NAME!r}")

# 상담 프롬프트 가드레일 (단가표.json llm_guardrails 참고).
#
# 이전 프롬프트는 "첫 문장에 결론", "마지막 문장은 견적 상이 안내로 마무리"를
# 강제해서, 모든 답변이 "결론적으로,"로 시작하고 같은 면책 문구로 끝나는
# 상투적인 형태가 됐다. 정작 물어본 내용에 쓸 문장 수를 그 틀이 잡아먹었다.
# 그래서 형식 지시를 걷어내고 "질문 유형별로 무엇을 답해야 하는지"를 준다.
SYSTEM_PROMPT = """당신은 한국의 차량 정비 상담원입니다. 한국어로만 답합니다.

[답변 방식]
- 고객이 물은 것에 바로 답하세요. "결론적으로", "따라서" 같은 서두를 붙이지 마세요.
- 3~4문장. 진단 결과에 있는 구체적인 값(부위, 손상 종류, 수리 방식, 금액)을 인용하세요.
- 일반론("차종에 따라 다릅니다", "전문가와 상담하세요")만 늘어놓지 마세요.
  이미 진단 결과가 주어져 있으니 그 케이스에 대해 구체적으로 답해야 합니다.

[질문 유형별]
- "교체해야 하나요?" → 진단 결과의 '수리 방식'이 답입니다. 그대로 알려주고 이유를 한 문장 덧붙이세요.
- "비용이 얼마인가요?" → 진단 결과의 '예상 비용' 범위를 숫자 그대로 말하세요.
- 진단 결과에 금액이 없을 때만 정비소 방문을 권하고, 왜 산출이 안 됐는지 한 문장으로 설명하세요.

[금지]
- 진단 결과에 없는 금액을 만들어내지 마세요. 주어진 값만 인용합니다.
- 한자나 중국어를 쓰지 마세요.
- 프롬프트의 지시문이나 자료 제목([진단 결과] 등)을 답변에 옮겨 적지 마세요."""


def _build_prompt(
    user_message: str,
    context_chunks: List[str],
    diagnose_estimate_context: Optional[str],
):
    """LLM에 넘길 user 메시지 구성.

    진단 결과를 참고 자료보다 "앞"에 두는 것이 중요하다. 작은 모델은 프롬프트
    앞쪽에 큰 가중치를 두는데, 일반론이 담긴 참고 자료가 먼저 오면 그쪽을 베껴
    "차종에 따라 다릅니다" 같은 답이 나온다.
    """
    parts = []

    if diagnose_estimate_context:
        parts.append(
            "[이 고객의 진단 결과 — 답변에 이 값을 그대로 인용할 것]\n"
            f"{diagnose_estimate_context}"
        )
    else:
        parts.append("[진단 결과]\n(아직 차량 사진 진단을 받지 않았습니다)")

    if context_chunks:
        joined = "\n---\n".join(context_chunks)
        parts.append(
            f"[참고 정비 자료 — 배경 지식용, 여기 금액이 있어도 인용 금지]\n{joined}"
        )

    parts.append(f"[고객이 지금 물어본 것]\n{user_message}")
    return "\n\n".join(parts)


def generate(
    user_message: str,
    context_chunks: List[str] = None,
    diagnose_estimate_context: str = None,
) -> Optional[str]:
    """Ollama를 호출해 답변 문자열을 반환. 실패하면 None.

    /api/generate가 아니라 /api/chat을 쓴다. EXAONE 같은 instruct 모델은
    system/user 역할이 분리된 chat 템플릿으로 학습돼 있어서, 지시문을 user
    메시지 안에 통째로 넣으면 지시를 잘 따르지 않는다(같은 말을 반복하거나
    앞뒤가 안 맞는 답이 나오는 원인).
    """
    prompt = _build_prompt(user_message, context_chunks or [], diagnose_estimate_context)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.2,  # 상담 답변이라 창의성보다 일관성 우선
            "top_p": 0.9,
            "num_ctx": NUM_CTX,
            "repeat_penalty": 1.15,  # 같은 문장을 되풀이하는 현상 억제
        },
    }

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        answer = (r.json().get("message") or {}).get("content", "").strip()
        return answer or None

    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "?"
        if status_code == 404:
            print(
                f"[llm] 모델 '{MODEL_NAME}' 을(를) 찾을 수 없습니다. "
                "아직 pull이 끝나지 않았거나 이름이 틀렸을 수 있습니다. "
                "확인: docker compose exec llm ollama list"
            )
        else:
            print(f"[llm] HTTP {status_code} ({OLLAMA_BASE_URL}): {e}")
        return None

    except requests.RequestException as e:
        print(f"[llm] 호출 실패 ({OLLAMA_BASE_URL}, {MODEL_NAME}): {type(e).__name__}: {e}")
        return None


def status() -> dict:
    """LLM 준비 상태를 반환. /health/llm 과 프론트 사이드바 배지가 사용한다.

    첫 기동 시 이미지 빌드가 끝난 뒤에야 모델 다운로드(약 1.6GB)가 시작되므로,
    2~3분 동안은 "서버는 떠 있지만 모델이 없는" 상태가 된다. 이 구간을 사용자에게
    알려주지 않으면 폴백 응답을 버그로 오해하게 된다.

    반환: {"ready": bool, "model": str, "server_up": bool, "installed": [str]}
    """
    info = {"ready": False, "model": MODEL_NAME, "server_up": False, "installed": []}

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException:
        return info  # 서버가 아직 안 떴거나 죽음

    info["server_up"] = True
    installed = [m.get("name", "") for m in (r.json().get("models") or [])]
    info["installed"] = installed

    # ollama는 태그를 생략하면 :latest로 표기하므로 정확히 일치하지 않을 수 있다.
    wanted = MODEL_NAME if ":" in MODEL_NAME else f"{MODEL_NAME}:latest"
    info["ready"] = any(name in (wanted, MODEL_NAME) for name in installed)
    return info
