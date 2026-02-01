from __future__ import annotations

import json
from dataclasses import dataclass

from backend.agents.base import BaseAgent
from backend.llm.router import ModelRouter
from backend.retrieval.context import ContextBundle


@dataclass(frozen=True)
class SupervisorDecision:
    specialists: list[str]
    rationale: str
    instructions: dict[str, str]


class SupervisorAgent(BaseAgent):
    name = "supervisor"
    system_prompt = (
        "You are the supervisor for a medical research assistant. "
        "This is general information, not medical advice. Only a clinician can diagnose or prescribe. "
        "If red flags appear, advise urgent evaluation. Use retrieved evidence; if none, say so.\n\n"
        "Decide which specialist agents to call. Allowed specialists: cardiology, geriatrics, mental. "
        "If unsure, call all three.\n\n"
        "Return VALID JSON ONLY with keys:\n"
        "- specialists: list of strings (subset of allowed)\n"
        "- rationale: string\n"
        "- instructions: object mapping specialist -> instruction\n"
        "No extra text."
    )

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def run(self, query: str, context: ContextBundle) -> SupervisorDecision:
        messages = self.build_messages(query, context)
        resp = await self._router.generate(
            role="supervisor",
            messages=messages,
            temperature=0.1,
            max_tokens=300,
        )
        decision = self._parse_decision(resp.content)
        if decision is None:
            repair_messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": "Return VALID JSON ONLY. No extra text, no markdown."},
            ]
            resp2 = await self._router.generate(
                role="supervisor",
                messages=repair_messages,
                temperature=0.0,
                max_tokens=200,
            )
            decision = self._parse_decision(resp2.content)
        return decision or SupervisorDecision(
            specialists=["cardiology", "geriatrics", "mental"],
            rationale="Failed to parse supervisor output; defaulting to all specialists.",
            instructions={},
        )

    @staticmethod
    def _parse_decision(text: str) -> SupervisorDecision | None:
        try:
            data = json.loads(_extract_json(text))
            raw_specialists = list(data.get("specialists", []))
            allowed = {"cardiology", "geriatrics", "mental"}
            specialists = [
                str(item).strip().lower()
                for item in raw_specialists
                if str(item).strip().lower() in allowed
            ]
            rationale = str(data.get("rationale", "")).strip()
            instructions = dict(data.get("instructions", {}))
            if not specialists:
                specialists = ["cardiology", "geriatrics", "mental"]
            return SupervisorDecision(
                specialists=specialists,
                rationale=rationale or "No rationale provided.",
                instructions=instructions,
            )
        except Exception:
            return None


def _extract_json(text: str) -> str:
    if "{" not in text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text
