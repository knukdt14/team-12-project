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
    """질문과 유사한 상위 k개 청크의 텍스트 리스트를 반환.

    인덱스가 없으면 빈 리스트를 반환한다 (에러를 던지지 않음 — 호출부에서 폴백 처리).
    """
    store = _load_vectorstore()
    if store is None:
        return []

    results = store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
