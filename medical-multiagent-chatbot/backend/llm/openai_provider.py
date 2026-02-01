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
            if resp.status_code == 404 and "localhost:11434" in self.base_url:
                return await self._ollama_chat(client, model_id, messages)
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

    async def _ollama_chat(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        messages: list[LLMMessage],
    ) -> LLMResponse:
        payload = {"model": model_id, "messages": messages, "stream": False}
        resp = await client.post(
            f"{self._ollama_base().rstrip('/')}/api/chat",
            json=payload,
        )
        if resp.status_code == 404:
            return await self._ollama_generate(client, model_id, messages)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(content=content, model=model_id, usage=None, raw=data)

    def _ollama_base(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url

    async def _ollama_generate(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        messages: list[LLMMessage],
    ) -> LLMResponse:
        prompt = _messages_to_prompt(messages)
        payload = {"model": model_id, "prompt": prompt, "stream": False}
        resp = await client.post(
            f"{self._ollama_base().rstrip('/')}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        content = data.get("response", "")
        return LLMResponse(content=content, model=model_id, usage=None, raw=data)


def _messages_to_prompt(messages: list[LLMMessage]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)

#this is making a real API call
