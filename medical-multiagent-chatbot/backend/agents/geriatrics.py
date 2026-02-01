from __future__ import annotations

import json

from backend.agents.base import AgentOutput, BaseAgent
from backend.llm.router import ModelRouter
from backend.retrieval.context import ContextBundle


class GeriatricsAgent(BaseAgent):
    name = "geriatrics"
    system_prompt = (
        "You are a geriatrics specialist for a medical research assistant. This is general information, not medical advice. Only a clinician can diagnose or prescribe. If red flags appear, advise urgent evaluation. Use retrieved evidence; if none, say so.\n"
        "\n"
        "Return VALID JSON ONLY with keys:\n"
        "summary (string), key_points (list of strings), red_flags (list of strings), uncertainties (list of strings), evidence_links (list of strings), draft_response_text (string).\n"
        "Rules:\n"
        "- Summary <= 80 words and must use labels with semicolons: 'Likely causes: ...; First checks: ...; Urgent red flags: ...'.\n"
        "- Each key point must be a short sentence plus an evidence ID in parentheses (not just the ID).\n"
        "- If no evidence, set evidence_links=['NO_EVIDENCE'], set key_points=[], and explicitly say no evidence in summary.\n"
        "- draft_response_text must be patient-facing and short: include headings exactly 'What might be going on:', 'What to check first:', 'When to seek urgent care:' with 1-3 bullets each; no evidence IDs.\n"
        "- Focus on: likely causes (3-5), first checks/workup steps (3-5), and urgent red flags.\n"
        "- Consider geriatric issues: delirium, falls risk, polypharmacy, dehydration, infection, hypoxia.\n"
        "- Do NOT mention medication dosing or changes. No diagnosis.\n"
        "- No extra text outside JSON."
    )

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def run(self, query: str, context: ContextBundle) -> AgentOutput:
        messages = self.build_messages(query, context)
        resp = await self._router.generate(
            role="geriatrics",
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        output = self._parse_output(resp)
        if _is_empty(output):
            repair_messages = messages + [
                {"role": "assistant", "content": resp.content},
                {
                    "role": "user",
                    "content": "Return VALID JSON ONLY. No extra text, no markdown.",
                },
            ]
            resp2 = await self._router.generate(
                role="geriatrics",
                messages=repair_messages,
                temperature=0.0,
                max_tokens=400,
            )
            output = self._parse_output(resp2)
        if _is_empty(output):
            draft_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY a short patient-facing response (no JSON). "
                        "Use headings exactly: 'What might be going on:', 'What to check first:', "
                        "'When to seek urgent care:' with 1-3 bullets each. "
                        "No evidence IDs, no medication changes, no diagnosis."
                    ),
                }
            ]
            resp3 = await self._router.generate(
                role="geriatrics",
                messages=draft_messages,
                temperature=0.2,
                max_tokens=250,
            )
            draft_text = resp3.content.strip()
            if not draft_text:
                draft_text = (
                    "What might be going on:\n"
                    "- Older adults can develop confusion or falls from several causes.\n\n"
                    "What to check first:\n"
                    "- Review recent changes and check vital signs with a clinician.\n\n"
                    "When to seek urgent care:\n"
                    "- Worsening breathlessness, new severe confusion, or repeated falls."
                )
            output = AgentOutput(
                summary="",
                key_points=[],
                red_flags=[],
                uncertainties=[],
                evidence_links=[],
                draft_response_text=draft_text,
                token_usage=_usage_from(resp3),
            )
        evidence_ids = self._evidence_ids(context)
        if evidence_ids and (not output.evidence_links or "NO_EVIDENCE" in output.evidence_links):
            repair_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Return VALID JSON ONLY. You must include evidence_links using at least one "
                        f"of these IDs: {', '.join(evidence_ids[:20])}."
                    ),
                },
            ]
            resp3 = await self._router.generate(
                role="geriatrics",
                messages=repair_messages,
                temperature=0.0,
                max_tokens=400,
            )
            output = self._parse_output(resp3)
        return output

    @staticmethod
    def _parse_output(resp) -> AgentOutput:
        usage = _usage_from(resp)
        content = _extract_json(resp.content)
        try:
            data = json.loads(content)
            return AgentOutput(
                summary=str(data.get("summary", "")).strip(),
                key_points=list(data.get("key_points", [])),
                red_flags=list(data.get("red_flags", [])),
                uncertainties=list(data.get("uncertainties", [])),
                evidence_links=list(data.get("evidence_links", [])),
                draft_response_text=str(data.get("draft_response_text", "")).strip(),
                token_usage=usage,
            )
        except Exception:
            return AgentOutput(
                summary="",
                key_points=[],
                red_flags=[],
                uncertainties=[],
                evidence_links=[],
                draft_response_text="",
                token_usage=usage,
            )


def _extract_json(text: str) -> str:
    if "{" not in text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text


def _is_empty(output: AgentOutput) -> bool:
    return not output.summary and not output.draft_response_text


def _usage_from(resp) -> dict[str, int] | None:
    usage_obj = getattr(resp, "usage", None)
    if not usage_obj:
        return None
    return {
        "prompt_tokens": usage_obj.prompt_tokens,
        "completion_tokens": usage_obj.completion_tokens,
        "total_tokens": usage_obj.total_tokens,
    }
