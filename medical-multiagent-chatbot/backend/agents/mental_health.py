from __future__ import annotations

import json

from backend.agents.base import AgentOutput, BaseAgent
from backend.llm.router import ModelRouter
from backend.retrieval.context import ContextBundle


class MentalHealthAgent(BaseAgent):
    name = "mental"
    system_prompt = (
        "You are a mental health specialist for a medical research assistant. "
        "This is general information, not medical advice. Only a clinician can diagnose or prescribe. "
        "If red flags appear, advise urgent evaluation. Use retrieved evidence; if none, say so. "
        "If there is any risk of self-harm, urge immediate professional help or emergency services.\n\n"
        "Return VALID JSON ONLY with keys:\n"
        "summary (string), key_points (list of strings), red_flags (list of strings), "
        "uncertainties (list of strings), evidence_links (list of strings), "
        "draft_response_text (string).\n"
        "Rules:\n"
        "- Cite evidence_links for every key point (KB IDs or PMIDs).\n"
        "- If no evidence, set evidence_links=[\"NO_EVIDENCE\"] and say so in summary.\n"
        "- Keep summary ≤ 80 words; key_points ≤ 5 items.\n"
        "- Do NOT mention medication dosing or changes.\n"
        "- No extra text outside JSON."
    )

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def run(self, query: str, context: ContextBundle) -> AgentOutput:
        messages = self.build_messages(query, context)
        resp = await self._router.generate(
            role="mental",
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        return self._parse_output(resp.content)

    @staticmethod
    def _parse_output(text: str) -> AgentOutput:
        try:
            data = json.loads(text)
            return AgentOutput(
                summary=str(data.get("summary", "")).strip(),
                key_points=list(data.get("key_points", [])),
                red_flags=list(data.get("red_flags", [])),
                uncertainties=list(data.get("uncertainties", [])),
                evidence_links=list(data.get("evidence_links", [])),
                draft_response_text=str(data.get("draft_response_text", "")).strip(),
            )
        except Exception:
            return AgentOutput(
                summary=text.strip(),
                key_points=[],
                red_flags=[],
                uncertainties=[],
                evidence_links=[],
                draft_response_text=text.strip(),
            )
