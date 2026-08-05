"""ai/docs/*.md -> 청킹 -> ko-sroberta 임베딩 -> FAISS 인덱스 저장.

실행 (프로젝트 루트 기준):
    python ai/build_vectorstore.py

README 스펙: chunk_size=600자, overlap=80자, jhgan/ko-sroberta-multitask 임베딩.
빌드 타임에 한 번 실행해서 backend/data/vectorstore/에 인덱스를 저장해두고,
런타임(services/rag.py)에서는 저장된 인덱스를 불러오기만 한다 (Docker 빌드 시
이 스크립트를 실행하도록 Dockerfile에 반영 예정 — 런타임 생성은 배포 지연 원인이라 금지).
"""
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent  # ai/ -> 프로젝트 루트
DOCS_DIR = BASE_DIR / "ai" / "docs"
VECTORSTORE_DIR = BASE_DIR / "backend" / "data" / "vectorstore"

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


def load_documents():
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    if not docs:
        raise SystemExit(f"{DOCS_DIR}에 markdown 문서가 없습니다. ai/docs/*.md를 먼저 작성하세요.")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # "## 소제목" 경계 우선 분리
    )
    return splitter.split_documents(docs)


def build_and_save(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def main():
    docs = load_documents()
    print(f"문서 {len(docs)}개 로드")

    chunks = split_documents(docs)
    print(f"청크 {len(chunks)}개 생성 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    build_and_save(chunks)
    print(f"FAISS 인덱스 저장 완료: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
