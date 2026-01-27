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
        return AgentOutput(
            summary=response.content.strip(),
            key_points=[],
            red_flags=[],
            uncertainties=[],
            evidence_links=[],
            draft_response_text=response.content.strip(),
        )

    def _format_context(self, context: ContextBundle) -> str:
        kb_lines = [
            f"[KB:{rec.doc_id}:{rec.chunk_id}] {rec.text}" for rec in context.kb_passages
        ]
        pm_lines = [
            f"[PMID:{ref.pmid}] {ref.title} ({ref.journal}, {ref.year})"
            for ref in context.pubmed_refs
        ]
        return "\n".join(kb_lines + pm_lines) if (kb_lines or pm_lines) else "No context."

"""
It standardizes how an agent turn question and retrieved context into LLM Messages
and turns the LLM Replyinto a structured output
"""