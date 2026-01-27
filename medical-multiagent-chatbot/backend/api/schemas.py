from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RoleName = Literal["default", "supervisor", "cardiology", "geriatrics", "mental"]


class LLMMessageIn(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class LLMTestRequest(BaseModel):
    role: RoleName = "default"
    messages: list[LLMMessageIn]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=2048)


class LLMUsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMTestResponse(BaseModel):
    content: str
    model: str
    usage: LLMUsageOut | None = None

#this is a smoke test endpoint to confirm if my llm api base url works, my api key works and my routing works from the role to model andthe request and response format is corect