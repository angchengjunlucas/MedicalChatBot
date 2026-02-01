from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.retrieval.vectorstore import VectorRecord
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class HybridResult:
    records: list[VectorRecord]


def keyword_score(query: str, text: str) -> float:
    q_terms = _tokenize(query)
    if not q_terms:
        return 0.0
    t_terms = _tokenize(text)
    if not t_terms:
        return 0.0
    hits = sum(1 for t in q_terms if t in t_terms)
    return hits / len(set(q_terms))


def merge_hybrid(
    query: str,
    dense: list[VectorRecord],
    keyword_pool: list[VectorRecord],
    top_k: int = 5,
) -> HybridResult:
    scored: dict[str, VectorRecord] = {}
    for rec in dense:
        scored[rec.chunk_id] = rec

    if keyword_pool:
        corpus_tokens = [_tokenize(rec.text) for rec in keyword_pool]
        bm25 = BM25Okapi(corpus_tokens)
        q_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(q_tokens) if q_tokens else []

        for idx, rec in enumerate(keyword_pool):
            bm25_score = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
            kw_score = keyword_score(query, rec.text)
            score = bm25_score + kw_score
            if score <= 0:
                continue
            boosted = VectorRecord(
                doc_id=rec.doc_id,
                chunk_id=rec.chunk_id,
                text=rec.text,
                metadata=rec.metadata,
                score=(rec.score or 0.0) + score,
            )
            scored[rec.chunk_id] = boosted

    ranked = sorted(scored.values(), key=lambda r: (r.score or 0.0), reverse=True)
    return HybridResult(records=ranked[:top_k])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())
