"""Ollama REST 호출 클라이언트 (기본 EXAONE-3.5-2.4B-Instruct, GGUF Q4).

Docker Compose 환경에서는 컨테이너 내부 네트워크로 http://llm:11434 를 쓰고,
로컬 단독 실행 시에는 http://localhost:11434 를 쓴다 — 환경변수 OLLAMA_BASE_URL로
오버라이드 가능. 사양이 부족하면 .env의 OLLAMA_MODEL만 바꾸면 된다
(예: OLLAMA_MODEL=qwen2.5:0.5b).

장애 처리: LLM이 아직 모델을 내려받는 중이거나 죽어 있어도 서비스 전체가
멈추면 안 되므로, generate()는 예외를 던지지 않고 None을 반환한다.
호출부(routers/chat.py)가 폴백 문구로 대체한다.
"""
import os

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b")
MAX_TOKENS = 300
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # 로컬 CPU 추론은 느릴 수 있어 여유있게


def generate(prompt: str, system: str = None):
    """Ollama /api/generate 호출해서 답변 문자열을 반환 (비스트리밍).

    연결 실패/타임아웃/모델 미준비 등 어떤 이유로든 실패하면 None을 반환한다.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": MAX_TOKENS},
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.RequestException as e:
        print(f"[llm_client] Ollama 호출 실패 — 폴백으로 넘어갑니다: {type(e).__name__}: {e}")
        return None
