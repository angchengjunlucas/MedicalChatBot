from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.llm.base import LLMMessage, LLMResponse
from backend.retrieval.context import ContextBundle


@dataclass(frozen=True)
class AgentOutput:
    summary: str
    key_points: list[str]
    red_flags: list[str]
    uncertainties: list[str]
    evidence_links: list[str]
    draft_response_text: str
    token_usage: dict[str, int] | None = None


class BaseAgent:
    name: str = "base"
    system_prompt: str = ""

    async def run(self, query: str, context: ContextBundle) -> AgentOutput:
        raise NotImplementedError

    def build_messages(self, query: str, context: ContextBundle) -> list[LLMMessage]:
        context_text = self._format_context(context)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"User question: {query}\n\nContext:\n{context_text}"},
        ]

    def parse_response(self, response: LLMResponse) -> AgentOutput:
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return AgentOutput(
            summary=response.content.strip(),
            key_points=[],
            red_flags=[],
            uncertainties=[],
            evidence_links=[],
            draft_response_text=response.content.strip(),
            token_usage=usage,
        )

    def _format_context(self, context: ContextBundle) -> str:
        kb_lines = [
            f"[KB:{rec.doc_id}:{rec.chunk_id}] {rec.text}" for rec in context.kb_passages
        ]
        pm_lines = [
            _format_pubmed(ref)
            for ref in context.pubmed_refs
        ]
        return "\n".join(kb_lines + pm_lines) if (kb_lines or pm_lines) else "No context."

    def _evidence_ids(self, context: ContextBundle) -> list[str]:
        ids: list[str] = []
        for rec in context.kb_passages:
            ids.append(f"KB:{rec.doc_id}:{rec.chunk_id}")
        for ref in context.pubmed_refs:
            ids.append(f"PMID:{ref.pmid}")
        return ids


def _format_pubmed(ref) -> str:
    base = f"[PMID:{ref.pmid}] {ref.title} ({ref.journal}, {ref.year})"
    if ref.abstract:
        return f"{base}\nAbstract: {ref.abstract}"
    return base

"""
It standardizes how an agent turn question and retrieved context into LLM Messages
and turns the LLM Replyinto a structured output
"""
