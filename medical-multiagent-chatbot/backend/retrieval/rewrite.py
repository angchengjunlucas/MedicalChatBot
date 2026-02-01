from __future__ import annotations

from backend.llm.base import LLMMessage
from backend.llm.router import ModelRouter
from backend.retrieval.prompts import QUERY_REWRITE_SYSTEM


class QueryRewriter:
    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def rewrite(self, query: str) -> str:
        messages: list[LLMMessage] = [
            {"role": "system", "content": QUERY_REWRITE_SYSTEM},
            {"role": "user", "content": query},
        ]
        resp = await self._router.generate(
            role="supervisor",
            messages=messages,
            temperature=0.1,
            max_tokens=64,
        )
        return resp.content.strip()

