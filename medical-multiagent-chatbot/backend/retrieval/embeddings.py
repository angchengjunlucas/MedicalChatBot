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
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        items = data.get("data", [])
        # Preserve ordering by index if provided
        indexed = sorted(items, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in indexed]

# this generates embeddings for the text which is stores in Chroma