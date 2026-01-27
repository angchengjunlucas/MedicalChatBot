from __future__ import annotations

from dataclasses import asdict, is_dataclass

from fastapi import APIRouter, HTTPException, Query

from backend.core.config import get_settings
from backend.retrieval.pubmed_client import PubMedClient

router = APIRouter()


@router.get("/pubmed_test")
async def pubmed_test(q: str = Query(..., min_length=2), retmax: int = 5):
    settings = get_settings()
    if not settings.pubmed_email or not settings.pubmed_tool_name:
        raise HTTPException(status_code=500, detail="PUBMED_EMAIL or PUBMED_TOOL_NAME not set")

    client = PubMedClient(
        email=settings.pubmed_email,
        tool=settings.pubmed_tool_name,
    )
    articles = await client.search_and_summary(q, retmax=retmax)
    serialized = []
    for item in articles:
        if is_dataclass(item):
            serialized.append(asdict(item))
        elif isinstance(item, dict):
            serialized.append(item)
        else:
            serialized.append(getattr(item, "__dict__", {"value": str(item)}))

    return {"count": len(articles), "articles": serialized}

# fastapi to test my pubmed retrieval code
