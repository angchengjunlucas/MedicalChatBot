from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.retrieval.pubmed_client import PubMedArticle
from backend.retrieval.vectorstore import VectorRecord


@dataclass(frozen=True)
class ContextBundle:
    query: str
    rewritten_query: str | None
    kb_passages: list[VectorRecord]
    pubmed_refs: list[PubMedArticle]
    notes: dict[str, Any]

#after retrieval from local KB and pubmed, we bundle everything together so that it can be used for the agent to answer