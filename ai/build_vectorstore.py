"""``ai/docs`` 정비 지식을 헤더 단위로 청킹해 FAISS 인덱스를 생성한다.

문서 제목·섹션·상대 경로를 각 청크 메타데이터에 보존하므로 검색 결과를
사용자에게 근거로 표시할 수 있다. 저장 디렉터리에는 문서 해시와 설정을 담은
manifest도 함께 기록해 어떤 원문으로 만든 인덱스인지 확인할 수 있다.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "ai" / "docs"
VECTORSTORE_DIR = BASE_DIR / "backend" / "data" / "vectorstore"

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
FRONT_MATTER_RE = re.compile(
    r"\A---\s*\r?\n(?P<meta>.*?)\r?\n---\s*(?:\r?\n|$)",
    re.DOTALL,
)


def parse_front_matter(text):
    match = FRONT_MATTER_RE.match(text)
    if match is None:
        return text, {}
    metadata = {}
    for line in match.group("meta").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return text[match.end():].lstrip(), metadata


def load_documents():
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=False,
    )

    documents = []
    source_hashes = {}
    for path in sorted(DOCS_DIR.glob("*.md"), key=lambda item: item.name):
        raw_text = path.read_text(encoding="utf-8")
        text, front_matter = parse_front_matter(raw_text)
        if not text.strip():
            continue

        relative_source = str(path.relative_to(BASE_DIR)).replace("\\", "/")
        source_hashes[relative_source] = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        for document in header_splitter.split_text(text):
            document.metadata.update(
                {
                    "source": relative_source,
                    "file_name": path.name,
                    "title": document.metadata.get(
                        "title", front_matter.get("title", path.stem)
                    ),
                    "section": document.metadata.get(
                        "subsection",
                        document.metadata.get("section", document.metadata.get("title", path.stem)),
                    ),
                    "keywords": " ".join(front_matter.values()),
                }
            )
            documents.append(document)

    if not documents:
        raise SystemExit(f"{DOCS_DIR}에 유효한 Markdown 문서가 없습니다.")
    return documents, source_hashes


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def build_and_save(chunks, source_hashes):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": len(source_hashes),
        "chunk_count": len(chunks),
        "source_sha256": source_hashes,
    }
    (VECTORSTORE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return vectorstore


def main():
    documents, source_hashes = load_documents()
    print(f"문서 {len(source_hashes)}개, 헤더 섹션 {len(documents)}개 로드")

    chunks = split_documents(documents)
    print(f"청크 {len(chunks)}개 생성 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    build_and_save(chunks, source_hashes)
    print(f"FAISS 인덱스와 manifest 저장 완료: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
