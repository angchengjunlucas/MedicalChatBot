from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.core.config import get_settings
from backend.retrieval.embeddings import OpenAIEmbeddingsProvider
from backend.retrieval.pipeline import KBRetrievalPipeline
from backend.retrieval.vectorstore import ChromaVectorStore

router = APIRouter()


@router.get("/kb_test")
async def kb_test(q: str = Query(..., min_length=2), top_k: int = 5):
    settings = get_settings()
    if settings.vector_db_type != "chroma":
        raise HTTPException(status_code=500, detail="Only chroma is supported in this milestone")
    if not settings.chroma_path:
        raise HTTPException(status_code=500, detail="CHROMA_PATH not set")
    if not settings.llm_api_base or not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM_API_BASE or LLM_API_KEY not set")
    if not settings.embedding_model:
        raise HTTPException(status_code=500, detail="EMBEDDING_MODEL not set")

    store = ChromaVectorStore(path=settings.chroma_path)
    embedder = OpenAIEmbeddingsProvider(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model_id=settings.embedding_model,
    )
    pipeline = KBRetrievalPipeline(vector_store=store, embedder=embedder)
    result = await pipeline.retrieve(q, top_k=top_k)
    return {"count": len(result.records), "records": [r.__dict__ for r in result.records]}

# test point to show my local KNB retrieval works end to end