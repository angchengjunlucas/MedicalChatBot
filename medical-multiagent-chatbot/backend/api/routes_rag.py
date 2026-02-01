from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.core.config import get_settings
from backend.retrieval.embeddings import OpenAIEmbeddingsProvider
from backend.retrieval.pipeline import KBRetrievalPipeline, RAGPipeline
from backend.retrieval.pubmed_client import PubMedClient
from backend.retrieval.vectorstore import ChromaVectorStore

router = APIRouter()


@router.get("/rag_test")
async def rag_test(q: str = Query(..., min_length=2), top_k: int = 3, pubmed_k: int = 3):
    settings = get_settings()
    if settings.vector_db_type != "chroma":
        raise HTTPException(status_code=500, detail="Only chroma is supported in this milestone")
    if not settings.chroma_path:
        raise HTTPException(status_code=500, detail="CHROMA_PATH not set")
    if not settings.llm_api_base or not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM_API_BASE or LLM_API_KEY not set")
    if not settings.embedding_model:
        raise HTTPException(status_code=500, detail="EMBEDDING_MODEL not set")
    if not settings.pubmed_email or not settings.pubmed_tool_name:
        raise HTTPException(status_code=500, detail="PUBMED_EMAIL or PUBMED_TOOL_NAME not set")

    store = ChromaVectorStore(path=settings.chroma_path)
    embedder = OpenAIEmbeddingsProvider(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model_id=settings.embedding_model,
    )
    kb = KBRetrievalPipeline(vector_store=store, embedder=embedder)
    pubmed = PubMedClient(email=settings.pubmed_email, tool=settings.pubmed_tool_name)
    rag = RAGPipeline(kb_pipeline=kb, pubmed_client=pubmed)
    bundle = await rag.retrieve(q, top_k=top_k, pubmed_k=pubmed_k)

    return {
        "query": bundle.query,
        "rewritten_query": bundle.rewritten_query,
        "kb_passages": [r.__dict__ for r in bundle.kb_passages],
        "pubmed_refs": [p.__dict__ for p in bundle.pubmed_refs],
    }

"""
Full Route for RAG
1. Give question
2. It returns relevant chunks from local kb (chroma) 
3. It returns the relevant pubmed paper metadata
4. Both bundled together
"""
