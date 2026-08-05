"""FAISS 검색 + 프롬프트 구성 (RAG).

ai/build_vectorstore.py로 미리 빌드해둔 FAISS 인덱스(backend/data/vectorstore/)를
불러와서, 사용자 질문과 유사한 top-3 청크를 검색한다.

주의: 인덱스가 없으면(아직 build_vectorstore.py를 실행 안 한 상태) search()가
빈 리스트를 반환한다 — routers/chat.py는 이 경우 RAG 없이 일반 LLM 응답으로
폴백해야 한다.

langchain_community/faiss는 requirements.txt에 추가했지만 아직 설치 안 된 환경도
있을 수 있어서, 모듈 최상단이 아니라 함수 내부에서 import한다 — 이렇게 해야 이 패키지가
없어도 main.py 임포트 자체(=/diagnose, /estimate 포함 앱 전체)는 죽지 않고, /chat을
실제로 호출할 때만 에러가 난다.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # services/ -> backend/ -> 루트
VECTORSTORE_DIR = BASE_DIR / "backend" / "data" / "vectorstore"

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
TOP_K = 3

_vectorstore = None
_embeddings = None


def _load_vectorstore():
    """서버 시작 시(또는 첫 요청 시) 한 번만 FAISS 인덱스를 로드해 전역 캐싱한다."""
    global _vectorstore, _embeddings

    if _vectorstore is not None:
        return _vectorstore

    if not VECTORSTORE_DIR.exists():
        # 아직 build_vectorstore.py를 실행하지 않은 상태 — RAG 없이 동작하도록 None 유지
        return None

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    _vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        _embeddings,
        allow_dangerous_deserialization=True,  # 우리가 직접 빌드한 신뢰 가능한 로컬 인덱스라 허용
    )
    return _vectorstore


def search(query: str, k: int = TOP_K):
    """질문과 유사한 상위 k개 청크를 반환. [{"text", "source"}] 형태.

    인덱스가 없으면 빈 리스트를 반환한다 (에러를 던지지 않음 — 호출부에서 폴백 처리).
    """
    store = _load_vectorstore()
    if store is None or not query.strip():
        return []

    results = store.similarity_search(query, k=k)
    return [
        {"text": doc.page_content, "source": doc.metadata.get("source", "unknown")}
        for doc in results
    ]


SYSTEM_PROMPT = """당신은 한국의 차량 정비 상담원입니다. 한국인 고객에게 한국어로만 답합니다.

- 반드시 한국어로만 쓰세요. 한자나 중국어를 단 한 글자도 쓰지 마세요.
- 주어진 진단 결과와 참고 자료에 있는 내용만 근거로 답하세요.
- 금액은 진단 결과에 적힌 값만 인용하세요. 없으면 정비소 방문을 권하세요.
- 지시문이나 자료를 그대로 옮겨 적지 말고, 질문에 대한 답만 3~4문장으로 쓰세요.
- 첫 문장에 결론을 쓰고, 마지막 문장은 실제 견적이 차종·업체에 따라 다르다는 안내로 끝내세요."""


def build_prompt(question: str, contexts, diagnosis_summary: str = "") -> str:
    """검색 결과 + 진단 요약을 합쳐 LLM에 넘길 user 메시지를 만든다.

    시스템 지시(SYSTEM_PROMPT)는 여기 넣지 않고 llm_client.generate(prompt,
    system=...)로 role을 분리한다 — instruct 모델이 지시를 "따를 대상"으로
    인식하게 하기 위함. 질문은 맨 마지막에 둔다(질문 뒤에 지시문을 붙이면
    모델이 그 지시문까지 답변 대상으로 착각해 프롬프트를 그대로 읊는 문제가 있었음).
    """
    if contexts:
        context_text = "\n\n".join(f"- {c['text'].strip()}" for c in contexts)
    else:
        context_text = "(관련 자료 없음)"

    diagnosis_block = diagnosis_summary.strip() or "(아직 차량 사진 진단을 받지 않았습니다)"

    return f"""아래는 이 고객의 AI 진단 결과와 참고할 정비 자료입니다.

■ 진단 결과
{diagnosis_block}

■ 참고 자료
{context_text}

■ 고객 질문
{question}"""
