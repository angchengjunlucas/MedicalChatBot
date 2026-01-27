from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .vectorstore import ChromaVectorStore


@dataclass(frozen=True)
class TextDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TextChunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def chunk_text(
    doc: TextDocument,
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[TextChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    text = _normalize_text(doc.text)
    if not text:
        return []

    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks: list[TextChunk] = []
    buffer = ""
    chunk_idx = 0

    def flush(buf: str) -> None:
        nonlocal chunk_idx
        if not buf.strip():
            return
        chunk_id = f"{doc.doc_id}:{chunk_idx}"
        meta = dict(doc.metadata)
        meta.update({"doc_id": doc.doc_id, "chunk_id": chunk_id})
        chunks.append(TextChunk(doc_id=doc.doc_id, chunk_id=chunk_id, text=buf, metadata=meta))
        chunk_idx += 1

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= max_chars:
            buffer = f"{buffer}\n{para}".strip()
        else:
            flush(buffer)
            if overlap > 0 and chunks:
                tail = chunks[-1].text[-overlap:]
                buffer = f"{tail}\n{para}".strip()
            else:
                buffer = para

    flush(buffer)
    return chunks


async def ingest_documents(
    docs: list[TextDocument],
    embedder: EmbeddingProvider,
    store: ChromaVectorStore,
    max_chars: int = 1000,
    overlap: int = 100,
    batch_size: int = 64,
) -> int:
    chunks: list[TextChunk] = []
    for doc in docs:
        chunks.extend(chunk_text(doc, max_chars=max_chars, overlap=overlap))

    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [c.metadata for c in chunks]

    total = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        embeddings = await embedder.embed(batch_texts)
        await store.add_texts(batch_ids, batch_texts, embeddings, batch_metas)
        total += len(batch_texts)
    return total


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join([line for line in lines if line])
