from __future__ import annotations

from fastapi import APIRouter, HTTPException

import uuid

from backend.api.schemas import (
    AgentOutputOut,
    ChatRequest,
    ChatResponse,
    LLMTestRequest,
    LLMTestResponse,
)
from backend.core.config import get_settings
from backend.llm.openai_provider import OpenAICompatibleProvider
from backend.llm.router import ModelRouter
from backend.orchestration.runner import Orchestrator
from backend.retrieval.embeddings import OpenAIEmbeddingsProvider
from backend.retrieval.pipeline import KBRetrievalPipeline, RAGPipeline
from backend.retrieval.pubmed_client import PubMedClient
from backend.retrieval.vectorstore import ChromaVectorStore
from backend.safety.postprocess import safety_postprocess

router = APIRouter()


@router.post("/llm_test", response_model=LLMTestResponse)
async def llm_test(payload: LLMTestRequest) -> LLMTestResponse:
    settings = get_settings()
    if not settings.llm_api_base or not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM_API_BASE or LLM_API_KEY not set")

    provider = OpenAICompatibleProvider(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
    )
    model_router = ModelRouter(
        provider=provider,
        default_model=settings.default_model,
        supervisor_model=settings.supervisor_model,
        cardiology_model=settings.cardiology_model,
        geriatrics_model=settings.geriatrics_model,
        mental_model=settings.mental_model,
    )

    response = await model_router.generate(
        role=payload.role,
        messages=[m.model_dump() for m in payload.messages],
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )

    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return LLMTestResponse(content=response.content, model=response.model, usage=usage)


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    settings = get_settings()
    if settings.vector_db_type != "chroma":
        raise HTTPException(status_code=500, detail="Only chroma is supported in this milestone")
    if not settings.llm_api_base or not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM_API_BASE or LLM_API_KEY not set")
    if not settings.embedding_model:
        raise HTTPException(status_code=500, detail="EMBEDDING_MODEL not set")
    if not settings.pubmed_email or not settings.pubmed_tool_name:
        raise HTTPException(status_code=500, detail="PUBMED_EMAIL or PUBMED_TOOL_NAME not set")

    provider = OpenAICompatibleProvider(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
    )
    router_model = ModelRouter(
        provider=provider,
        default_model=settings.default_model,
        supervisor_model=settings.supervisor_model,
        cardiology_model=settings.cardiology_model,
        geriatrics_model=settings.geriatrics_model,
        mental_model=settings.mental_model,
    )

    store = ChromaVectorStore(path=settings.chroma_path)
    embedder = OpenAIEmbeddingsProvider(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model_id=settings.embedding_model,
    )
    kb = KBRetrievalPipeline(vector_store=store, embedder=embedder)
    pubmed = PubMedClient(email=settings.pubmed_email, tool=settings.pubmed_tool_name)
    rag = RAGPipeline(kb_pipeline=kb, pubmed_client=pubmed)
    orchestrator = Orchestrator(rag=rag, router=router_model)

    result = await orchestrator.run(payload.query)

    agents = {
        name: AgentOutputOut(**output.__dict__) for name, output in result.outputs.items()
    }
    response_text = "\n\n".join([o.draft_response_text for o in result.outputs.values()])
    safe = safety_postprocess(response_text)

    return ChatResponse(
        trace_id=str(uuid.uuid4()),
        response_text=safe.text,
        agents=agents,
    )

#this show taht my FASTAPI route works, my config/env variable are loaded correctly and my backend can actually call an appropriate model
