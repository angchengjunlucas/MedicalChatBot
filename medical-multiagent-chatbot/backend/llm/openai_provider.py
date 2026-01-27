from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .base import LLMMessage, LLMProvider, LLMResponse, LLMUsage


@dataclass(frozen=True)
class OpenAICompatibleProvider(LLMProvider):
    base_url: str
    api_key: str
    timeout_s: float = 30.0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        model_id: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = LLMUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return LLMResponse(content=content, model=data.get("model", model_id), usage=usage, raw=data)

#this is making a real API call