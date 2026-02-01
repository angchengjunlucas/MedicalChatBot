from __future__ import annotations

from dataclasses import dataclass

from backend.retrieval.context import ContextBundle
from backend.retrieval.hybrid import merge_hybrid
from backend.retrieval.embeddings import OpenAIEmbeddingsProvider
from backend.retrieval.pubmed_client import PubMedClient, PubMedArticle
from backend.retrieval.rewrite import QueryRewriter
from backend.retrieval.vectorstore import ChromaVectorStore, VectorRecord


@dataclass(frozen=True)
class KBRetrievalResult:
    records: list[VectorRecord]


class KBRetrievalPipeline:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: OpenAIEmbeddingsProvider,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5) -> KBRetrievalResult:
        embeddings = await self._embedder.embed([query])
        query_emb = embeddings[0]
        records = await self._store.query(query_emb, top_k=top_k)
        return KBRetrievalResult(records=records)

    async def retrieve_hybrid(self, query: str, top_k: int = 5) -> KBRetrievalResult:
        dense = await self.retrieve(query, top_k=top_k * 3)
        hybrid = merge_hybrid(query, dense.records, dense.records, top_k=top_k)
        return KBRetrievalResult(records=hybrid.records)


class RAGPipeline:
    def __init__(
        self,
        kb_pipeline: KBRetrievalPipeline,
        pubmed_client: PubMedClient,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._kb = kb_pipeline
        self._pubmed = pubmed_client
        self._rewriter = rewriter

    async def retrieve(
        self,
        query: str,
        top_k: int = 8,
        pubmed_k: int = 8,
    ) -> ContextBundle:
        rewritten = None
        if self._rewriter:
            try:
                rewritten = await self._rewriter.rewrite(query)
            except Exception:
                rewritten = None

        effective_query = rewritten or query
        kb_result = await self._kb.retrieve_hybrid(effective_query, top_k=top_k)
        pubmed_refs: list[PubMedArticle] = await self._pubmed.search_and_summary(
            effective_query,
            retmax=pubmed_k,
            include_abstracts=True,
        )
        return ContextBundle(
            query=query,
            rewritten_query=rewritten,
            kb_passages=kb_result.records,
            pubmed_refs=pubmed_refs,
            notes={},
        )

#this is the search my local knowledge base pipeline, where it takes the question, turns it into embedding and search chroma for the most similar stored chunks and returns those chunks
