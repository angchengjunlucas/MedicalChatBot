from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas import LLMTestRequest, LLMTestResponse
from backend.core.config import get_settings
from backend.llm.openai_provider import OpenAICompatibleProvider
from backend.llm.router import ModelRouter

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

#this show taht my FASTAPI route works, my config/env variable are loaded correctly and my backend can actually call an appropriate model