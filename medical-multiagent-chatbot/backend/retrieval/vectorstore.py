from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import chromadb


@dataclass(frozen=True)
class VectorRecord:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float | None = None


class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str = "kb_store") -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_texts(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_records(result)

    @staticmethod
    def _to_records(result: dict[str, Any]) -> list[VectorRecord]:
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        records: list[VectorRecord] = []
        for idx, doc_text in enumerate(docs):
            meta = metas[idx] if idx < len(metas) else {}
            doc_id = str(meta.get("doc_id", ""))
            chunk_id = str(meta.get("chunk_id", ids[idx] if idx < len(ids) else idx))
            dist = dists[idx] if idx < len(dists) else None
            score = (1.0 - dist) if isinstance(dist, (int, float)) else None
            records.append(
                VectorRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=doc_text,
                    metadata=meta or {},
                    score=score,
                )
            )
        return records

# own curated knowledge base locally, results in less hallucinations, better consistency, and better specialist behaviour
