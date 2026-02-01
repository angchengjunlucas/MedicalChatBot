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
from backend.llm.local_provider import LocalPeftProvider
from backend.llm.router import ModelRouter
from backend.orchestration.runner import Orchestrator
from backend.retrieval.embeddings import OpenAIEmbeddingsProvider
from backend.retrieval.pipeline import KBRetrievalPipeline, RAGPipeline
from backend.retrieval.pubmed_client import PubMedClient
from backend.retrieval.rewrite import QueryRewriter
from backend.retrieval.vectorstore import ChromaVectorStore
from backend.safety.postprocess import safety_postprocess
from backend.storage.db import init_db
from backend.storage.models import TraceRecord
from backend.storage.repo import now_utc_iso, save_trace

router = APIRouter()
_LOCAL_PROVIDER_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], LocalPeftProvider] = {}


@router.post("/llm_test", response_model=LLMTestResponse)
async def llm_test(payload: LLMTestRequest) -> LLMTestResponse:
    settings = get_settings()
    if not settings.local_base_model and (not settings.llm_api_base or not settings.llm_api_key):
        raise HTTPException(status_code=500, detail="LLM_API_BASE or LLM_API_KEY not set")

    provider = _build_provider(settings)
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
    if not settings.local_base_model and (not settings.llm_api_base or not settings.llm_api_key):
        raise HTTPException(status_code=500, detail="LLM_API_BASE or LLM_API_KEY not set")
    if not settings.embedding_model:
        raise HTTPException(status_code=500, detail="EMBEDDING_MODEL not set")
    if not settings.pubmed_email or not settings.pubmed_tool_name:
        raise HTTPException(status_code=500, detail="PUBMED_EMAIL or PUBMED_TOOL_NAME not set")

    provider = _build_provider(settings)
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
    rewriter = QueryRewriter(router_model)
    rag = RAGPipeline(kb_pipeline=kb, pubmed_client=pubmed, rewriter=rewriter)
    orchestrator = Orchestrator(rag=rag, router=router_model)

    result = await orchestrator.run(payload.query)

    agents = {
        name: AgentOutputOut(**output.__dict__) for name, output in result.outputs.items()
    }
    response_text = result.final_text
    safe_short = safety_postprocess(response_text)
    safe_full = safety_postprocess(result.full_text if hasattr(result, "full_text") else response_text)
    trace_id = str(uuid.uuid4())

    init_db(settings.log_db_url)
    token_usage = _sum_token_usage(result.outputs)
    save_trace(
        settings.log_db_url,
        TraceRecord(
            trace_id=trace_id,
            user_id=payload.user_id,
            query=payload.query,
            created_at=now_utc_iso(),
            response_text=safe_short.text,
            response_detail=safe_full.text,
            context_meta=result.context_meta,
            agent_outputs={name: output.__dict__ for name, output in result.outputs.items()},
            safety_flags=list({*safe_short.flags, *safe_full.flags}),
            token_usage=token_usage,
        ),
    )

    return ChatResponse(
        trace_id=trace_id,
        response_text=safe_short.text,
        response_detail=safe_full.text,
        agents=agents,
    )


def _sum_token_usage(outputs: dict[str, object]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for output in outputs.values():
        usage = getattr(output, "token_usage", None)
        if not usage:
            continue
        totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        totals["completion_tokens"] += int(usage.get("completion_tokens", 0))
        totals["total_tokens"] += int(usage.get("total_tokens", 0))
    return totals


def _build_provider(settings) -> OpenAICompatibleProvider | LocalPeftProvider:
    if settings.local_base_model:
        adapters = {
            "supervisor": settings.local_supervisor_adapter,
            "cardiology": settings.local_cardiology_adapter,
            "geriatrics": settings.local_geriatrics_adapter,
            "mental": settings.local_mental_adapter,
        }
        adapter_paths = {role: path for role, path in adapters.items() if path}
        cache_key = (settings.local_base_model, tuple(sorted(adapter_paths.items())))
        provider = _LOCAL_PROVIDER_CACHE.get(cache_key)
        if provider is None:
            provider = LocalPeftProvider(
                base_model_id=settings.local_base_model,
                adapter_paths=adapter_paths,
            )
            _LOCAL_PROVIDER_CACHE[cache_key] = provider
        return provider
    return OpenAICompatibleProvider(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
    )

#this show taht my FASTAPI route works, my config/env variable are loaded correctly and my backend can actually call an appropriate model
