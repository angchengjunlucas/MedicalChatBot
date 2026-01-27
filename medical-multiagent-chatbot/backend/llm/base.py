from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, TypedDict


Role = Literal["system", "user", "assistant", "tool"]


class LLMMessage(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: LLMUsage | None
    raw: dict[str, Any] | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        model_id: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError

# this is a nice interface for any language model as well as data types