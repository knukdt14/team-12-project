"""차량 정비 지식 검색.

FAISS 인덱스가 있으면 의미 검색을 사용하고, 인덱스·임베딩 모델이 없거나 로드에
실패해도 ``ai/docs`` 원문을 대상으로 한 한국어 키워드 검색을 항상 제공한다.
따라서 로컬 실행이나 네트워크가 제한된 Docker 빌드에서도 RAG가 조용히 꺼지지
않는다. 검색어에는 현재 질문뿐 아니라 진단·견적과 최근 대화도 함께 반영한다.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = BASE_DIR / "ai" / "docs"
VECTORSTORE_DIR = BASE_DIR / "backend" / "data" / "vectorstore"

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1800"))
LEXICAL_MIN_SCORE = float(os.getenv("RAG_LEXICAL_MIN_SCORE", "1.5"))

_vectorstore = None
_embeddings = None
_vectorstore_attempted = False
_vectorstore_error: Optional[str] = None
_load_lock = threading.Lock()

_WORD_RE = re.compile(r"[가-힣]{2,}|[a-zA-Z][a-zA-Z0-9_-]+")
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\r?\n(?P<meta>.*?)\r?\n---\s*(?:\r?\n|$)",
    re.DOTALL,
)

_STOPWORDS = {
    "관련", "경우", "그리고", "그러면", "그럼", "대한", "어떻게", "얼마",
    "있나요", "인가요", "해주세요", "차량", "자동차", "손상", "수리", "지금",
    "정도", "필요", "사진", "이번", "해야", "하는", "있는", "없는", "기준",
    "고객", "결과", "사용자", "선택값", "탐지", "신뢰도", "예상", "비용",
    "minor", "moderate", "severe",
}

_ALIASES = {
    "교체": ("교환",),
    "갈아": ("교환",),
    "찌그러짐": ("덴트", "dent"),
    "찍힘": ("찌그러짐", "덴트", "dent"),
    "문콕": ("찌그러짐", "덴트", "dent"),
    "찌그러졌": ("찌그러짐", "덴트", "dent"),
    "덴트": ("찌그러짐", "dent"),
    "기스": ("스크래치", "scratch"),
    "긁힘": ("스크래치", "scratch"),
    "긁혔": ("긁힘", "스크래치", "scratch"),
    "긁힌": ("긁힘", "스크래치", "scratch"),
    "긁었": ("긁힘", "스크래치", "scratch"),
    "찍었": ("찍힘", "찌그러짐", "덴트", "dent"),
    "금": ("균열", "크랙", "crack"),
    "크랙": ("균열", "crack"),
    "갈라졌": ("균열", "크랙", "crack"),
    "깨짐": ("파손", "broken", "shatter"),
    "깨졌": ("깨짐", "파손", "broken", "shatter"),
    "부서졌": ("깨짐", "파손", "broken", "shatter"),
    "앞범퍼": ("전방범퍼", "front_bumper", "front-bumper"),
    "뒷범퍼": ("후방범퍼", "rear_bumper", "rear-bumper"),
    "본넷": ("보닛", "후드", "hood", "bonnet"),
    "보닛": ("본넷", "후드", "hood", "bonnet"),
    "휀더": ("펜더", "fender"),
    "펜더": ("휀더", "fender"),
    "문짝": ("도어", "door"),
    "앞유리": ("전면유리", "윈드쉴드", "windshield"),
    "전조등": ("헤드램프", "헤드라이트", "headlamp"),
    "후미등": ("테일램프", "테일라이트", "taillamp"),
    "사이드미러": ("측면거울", "side_mirror"),
    "타이어": ("펑크", "tire_flat"),
    "보험": ("자차", "자기차량손해"),
}

# 한 글자 별칭은 일반 부분 문자열로 비교하면 `금액`, `보험금`, `요금`, `금요일`도
# 모두 유리의 금(crack)으로 오인한다. 실제 균열 표현으로 쓰이는 형태만 허용한다.
_SHORT_ALIAS_FORMS = {
    "금": {"금", "금이", "금은", "금도", "금만", "금간", "금갔"},
}

_JOSA = (
    "으로부터", "에게서", "에서는", "으로는", "이라면", "라면", "에서", "에게",
    "으로", "까지", "부터", "처럼", "보다", "하고", "이며", "이면", "라도",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "과", "와", "도",
)


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    title: str
    section: str
    source: str
    keywords: str = ""
    score: float = 0.0

    def __str__(self) -> str:
        return self.content

    def source_payload(self) -> dict:
        return {
            "title": self.title,
            "section": self.section,
            "source": self.source,
        }


def _clean_term(term: str) -> str:
    value = term.lower().strip("-_ ")
    for suffix in _JOSA:
        if value.endswith(suffix) and len(value) - len(suffix) >= 2:
            value = value[: -len(suffix)]
            break
    return value


def _query_terms(text: str) -> list[str]:
    terms = []
    for raw in _WORD_RE.findall(text.lower()):
        term = _clean_term(raw)
        if not term or term in _STOPWORDS:
            continue

        expanded = []
        for alias_key, aliases in _ALIASES.items():
            if len(alias_key) == 1:
                forms = _SHORT_ALIAS_FORMS.get(alias_key, {alias_key})
                matched = term in forms or any(
                    term.startswith(prefix)
                    for prefix in (f"{alias_key}갔", f"{alias_key}간")
                )
            else:
                matched = alias_key in term or term in alias_key
            if matched:
                expanded.extend(aliases)

        # 한 글자 원문은 검색 노이즈가 크므로 별칭 확장 결과만 사용한다.
        if len(term) >= 2:
            terms.append(term)
        terms.extend(expanded)
    return list(dict.fromkeys(terms))


def _section_intent_bonus(chunk: RetrievedChunk, query: str) -> float:
    """질문의 목적과 직접 맞는 근거 섹션이 청크 제한 전에 선택되도록 보강한다."""
    query_lower = query.lower()
    heading = f"{chunk.title} {chunk.section}".lower()
    searchable_metadata = f"{chunk.title} {chunk.section} {chunk.keywords}".lower()
    bonus = 0.0

    generic_price_terms = {"금액", "가격", "비용", "얼마", "추가비"}
    price_tokens = generic_price_terms | {"할증", "공임", "계수"}
    price_intent = any(token in query_lower for token in price_tokens)
    if price_intent:
        topic_terms = {
            term
            for term in _query_terms(query)
            if term not in generic_price_terms and term not in {"추가", "알려줘"}
        }
        topic_match = any(term in searchable_metadata for term in topic_terms)
        if "조정값" in heading and topic_match:
            # 특수색·수입차처럼 기본 단가에 없는 조정 규칙은 일반 패널 금액
            # 섹션보다 먼저 제공해야 임의의 다른 부품 가격을 답하지 않는다.
            bonus += 18.0
        elif topic_match and any(
            token in heading for token in ("금액", "가격", "비용")
        ):
            bonus += 12.0
        elif topic_match and "룰베이스" in heading:
            bonus += 6.0

    decision_intent = any(
        token in query_lower
        for token in ("교환", "교체", "pdr", "판금", "복원", "severe", "심각")
    )
    if decision_intent and any(
        token in heading for token in ("선택 기준", "수리 방식", "교환 판단", "복원 판단")
    ):
        bonus += 12.0

    return bonus


def _split_long_section(text: str, max_chars: int = 1500) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces, current = [], []
    current_len = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and current_len + len(paragraph) + 2 > max_chars:
            pieces.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _parse_front_matter(text: str) -> tuple[str, dict[str, str]]:
    """검색에 필요한 단순 ``key: value`` 메타데이터를 본문과 분리한다."""
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        return text, {}

    metadata = {}
    for line in match.group("meta").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return text[match.end():].lstrip(), metadata


@lru_cache(maxsize=1)
def _load_lexical_chunks() -> tuple[RetrievedChunk, ...]:
    chunks = []
    if not DOCS_DIR.exists():
        return tuple()

    for path in sorted(DOCS_DIR.glob("*.md"), key=lambda item: item.name):
        text, metadata = _parse_front_matter(path.read_text(encoding="utf-8"))
        title = metadata.get("title") or path.stem
        section = title
        buffer = []
        keywords = " ".join(metadata.values())

        def flush() -> None:
            nonlocal buffer
            body = "\n".join(buffer).strip()
            if not body:
                buffer = []
                return
            for piece in _split_long_section(body):
                chunks.append(
                    RetrievedChunk(
                        content=piece,
                        title=title,
                        section=section,
                        source=f"ai/docs/{path.name}",
                        keywords=keywords,
                    )
                )
            buffer = []

        for line in text.splitlines():
            header = _HEADER_RE.match(line)
            if header:
                level, heading = len(header.group(1)), header.group(2).strip()
                if level == 1:
                    title = heading
                    section = heading
                    continue
                flush()
                section = heading
                buffer.append(line)
            else:
                buffer.append(line)
        flush()

    return tuple(chunks)


def _load_vectorstore():
    global _vectorstore, _embeddings, _vectorstore_attempted, _vectorstore_error

    if _vectorstore is not None or _vectorstore_attempted:
        return _vectorstore

    with _load_lock:
        if _vectorstore is not None or _vectorstore_attempted:
            return _vectorstore
        _vectorstore_attempted = True

        if not VECTORSTORE_DIR.exists():
            _vectorstore_error = "vectorstore_not_found"
            return None

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS

            _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            _vectorstore = FAISS.load_local(
                str(VECTORSTORE_DIR),
                _embeddings,
                allow_dangerous_deserialization=True,
            )
            _vectorstore_error = None
        except Exception as exc:  # RAG 장애가 /chat 전체를 500으로 만들지 않게 한다.
            _vectorstore_error = type(exc).__name__
            print(f"[rag] FAISS 로드 실패, 키워드 검색으로 대체: {type(exc).__name__}")
        return _vectorstore


def _lexical_score(chunk: RetrievedChunk, terms: Sequence[str]) -> float:
    title = chunk.title.lower()
    section = chunk.section.lower()
    content = chunk.content.lower()
    keywords = chunk.keywords.lower()
    score = 0.0

    for term in terms:
        needle = term.lower()
        if not needle:
            continue
        if needle in title:
            score += 7.0
        if needle in section:
            score += 4.0
        if needle in keywords:
            score += 3.0
        count = content.count(term.lower())
        if count:
            score += min(count, 3) * 1.15
    return score


def _semantic_candidates(query: str, candidate_count: int) -> list[RetrievedChunk]:
    store = _load_vectorstore()
    if store is None:
        return []

    try:
        results = store.similarity_search_with_score(query, k=candidate_count)
    except Exception as exc:
        print(f"[rag] FAISS 검색 실패, 키워드 검색으로 대체: {type(exc).__name__}")
        return []

    chunks = []
    for document, distance in results:
        metadata = document.metadata or {}
        source = str(metadata.get("source", "ai/docs/unknown.md")).replace("\\", "/")
        if source.startswith(str(BASE_DIR).replace("\\", "/")):
            source = str(Path(source).relative_to(BASE_DIR)).replace("\\", "/")
        title = str(metadata.get("title") or Path(source).stem)
        section = str(metadata.get("section") or metadata.get("header") or title)
        semantic_score = 5.0 / (1.0 + max(float(distance), 0.0))
        content, front_matter = _parse_front_matter(document.page_content.strip())
        if not content:
            continue
        chunks.append(
            RetrievedChunk(
                content=content,
                title=title,
                section=section,
                source=source,
                keywords=str(
                    metadata.get("keywords") or " ".join(front_matter.values())
                ),
                score=semantic_score,
            )
        )
    return chunks


def _history_text(history: Optional[Sequence[Mapping[str, str]]]) -> str:
    if not history:
        return ""
    rows = []
    for item in history[-4:]:
        if item.get("role") != "user":
            continue
        content = str(item.get("content", "")).strip()
        if content:
            rows.append(content[:500])
    return " ".join(rows)


def search(
    query: str,
    k: int = TOP_K,
    diagnosis_context: str = "",
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> list[RetrievedChunk]:
    """질문·진단·최근 대화를 함께 사용해 관련 근거 청크를 반환한다."""
    combined_query = "\n".join(
        part for part in (query.strip(), diagnosis_context.strip(), _history_text(history)) if part
    )
    if not combined_query:
        return []

    terms = _query_terms(combined_query)
    if not terms:
        return []

    lexical = []
    has_domain_signal = False
    for chunk in _load_lexical_chunks():
        score = _lexical_score(chunk, terms)
        has_domain_signal = has_domain_signal or score > 0
        if score >= LEXICAL_MIN_SCORE:
            lexical.append(replace(chunk, score=score))

    # 의미 검색은 어떤 정비 문서와도 어휘 접점이 있을 때만 보강한다. 그렇지 않으면
    # 날씨·인사 같은 범위 밖 질문에도 FAISS가 억지로 상위 문서를 반환할 수 있다.
    semantic = (
        _semantic_candidates(combined_query, max(k * 3, 8))
        if has_domain_signal
        else []
    )

    # 같은 문서/섹션/본문은 두 검색 경로의 점수를 합쳐 재정렬한다.
    merged: dict[tuple[str, str, str], RetrievedChunk] = {}
    for chunk in lexical + semantic:
        key = (chunk.source, chunk.section, chunk.content[:160])
        previous = merged.get(key)
        if previous is None:
            merged[key] = chunk
        else:
            merged[key] = replace(previous, score=previous.score + chunk.score)

    reranked = [
        replace(chunk, score=chunk.score + _section_intent_bonus(chunk, combined_query))
        for chunk in merged.values()
    ]
    ranked = sorted(
        reranked,
        key=lambda item: (-item.score, item.source, item.section),
    )

    selected = []
    per_source: dict[str, int] = {}
    char_count = 0
    for chunk in ranked:
        if per_source.get(chunk.source, 0) >= 2:
            continue
        if selected and char_count + len(chunk.content) > MAX_CONTEXT_CHARS:
            continue
        selected.append(chunk)
        per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
        char_count += len(chunk.content)
        if len(selected) >= max(1, k):
            break
    return selected


def status() -> dict:
    """비용 없이 확인 가능한 RAG 준비 상태를 반환한다."""
    chunks = _load_lexical_chunks()
    vectorstore_present = VECTORSTORE_DIR.exists()
    if _vectorstore is not None:
        mode = "hybrid"
    elif vectorstore_present and not _vectorstore_attempted:
        # 첫 검색에서 지연 로드한다. 아직 메모리에 없다는 이유만으로 UI가
        # 키워드 전용 폴백이라고 오해하지 않도록 준비 상태를 구분한다.
        mode = "hybrid_pending"
    else:
        mode = "lexical_fallback"
    return {
        "ready": bool(chunks),
        "mode": mode,
        "documents": len({chunk.source for chunk in chunks}),
        "chunks": len(chunks),
        "vectorstore_present": vectorstore_present,
        "semantic_ready": _vectorstore is not None,
        "semantic_error": _vectorstore_error,
    }
