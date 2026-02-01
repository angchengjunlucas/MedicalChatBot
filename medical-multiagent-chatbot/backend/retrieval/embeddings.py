from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class OpenAIEmbeddingsProvider:
    base_url: str
    api_key: str
    model_id: str
    timeout_s: float = 30.0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {"model": self.model_id, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers=self._headers(),
                json=payload,
            )
            if resp.status_code in {400, 404} and "localhost:11434" in self.base_url:
                return await self._ollama_embed(client, texts)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        items = data.get("data", [])
        # Preserve ordering by index if provided
        indexed = sorted(items, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in indexed]

    async def _ollama_embed(self, client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            payload = {"model": self.model_id, "prompt": _truncate(text, 2000)}
            last_exc: Exception | None = None
            for attempt in range(4):
                try:
                    resp = await client.post(
                        f"{self._ollama_base().rstrip('/')}/api/embeddings",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data: dict[str, Any] = resp.json()
                    emb = data.get("embedding")
                    if emb is None:
                        raise RuntimeError("Ollama embeddings response missing 'embedding'")
                    embeddings.append(emb)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    await asyncio.sleep(0.5 * (2**attempt))
            if last_exc:
                raise last_exc
        return embeddings

    def _ollama_base(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# this generates embeddings for the text which is stores in Chroma
