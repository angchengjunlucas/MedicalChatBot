from __future__ import annotations

import asyncio
from dataclasses import dataclass

from backend.agents.base import AgentOutput
from backend.agents.cardiology import CardiologyAgent
from backend.agents.geriatrics import GeriatricsAgent
from backend.agents.mental_health import MentalHealthAgent
from backend.agents.supervisor import SupervisorAgent
from backend.llm.router import ModelRouter
from backend.orchestration.composer import FinalComposer
from backend.retrieval.context import ContextBundle
from backend.retrieval.pipeline import RAGPipeline


@dataclass(frozen=True)
class OrchestrationResult:
    supervisor_rationale: str
    outputs: dict[str, AgentOutput]
    final_text: str
    full_text: str
    context_meta: dict[str, object]


class Orchestrator:
    def __init__(
        self,
        rag: RAGPipeline,
        router: ModelRouter,
    ) -> None:
        self._rag = rag
        self._router = router

    async def run(self, query: str) -> OrchestrationResult:
        context = await self._rag.retrieve(query)
        supervisor = SupervisorAgent(self._router)
        decision = await supervisor.run(query, context)

        tasks = []
        if "cardiology" in decision.specialists:
            tasks.append(self._run_agent(CardiologyAgent(self._router), query, context))
        if "geriatrics" in decision.specialists:
            tasks.append(self._run_agent(GeriatricsAgent(self._router), query, context))
        if "mental" in decision.specialists:
            tasks.append(self._run_agent(MentalHealthAgent(self._router), query, context))

        results = await asyncio.gather(*tasks)
        outputs = {name: output for name, output in results}
        outputs = _filter_all_evidence(outputs, context)
        composed = FinalComposer().compose(query, outputs, decision.rationale)
        context_meta = {
            "rewritten_query": context.rewritten_query,
            "kb_count": len(context.kb_passages),
            "pubmed_count": len(context.pubmed_refs),
        }
        return OrchestrationResult(
            supervisor_rationale=decision.rationale,
            outputs=outputs,
            final_text=composed.short_text,
            full_text=composed.full_text,
            context_meta=context_meta,
        )

    @staticmethod
    async def _run_agent(agent, query: str, context: ContextBundle):
        output = await agent.run(query, context)
        return agent.name, output


def _filter_all_evidence(outputs: dict[str, AgentOutput], context: ContextBundle) -> dict[str, AgentOutput]:
    allowed = _allowed_evidence_ids(context)
    filtered: dict[str, AgentOutput] = {}
    for name, output in outputs.items():
        ev = [e for e in output.evidence_links if e in allowed]
        if not ev:
            if allowed:
                ev = sorted(allowed)[:3]
            else:
                ev = ["NO_EVIDENCE"]
        filtered[name] = AgentOutput(
            summary=output.summary,
            key_points=output.key_points,
            red_flags=output.red_flags,
            uncertainties=output.uncertainties,
            evidence_links=ev,
            draft_response_text=output.draft_response_text,
            token_usage=output.token_usage,
        )
    return filtered


def _allowed_evidence_ids(context: ContextBundle) -> set[str]:
    allowed: set[str] = set()
    for rec in context.kb_passages:
        allowed.add(f"KB:{rec.doc_id}:{rec.chunk_id}")
        allowed.add(rec.chunk_id)
    for ref in context.pubmed_refs:
        allowed.add(f"PMID:{ref.pmid}")
        allowed.add(ref.pmid)
    return allowed

"""
This manager runs the whole chatbot flow
It does three steps:
1. RAG
2. Ask supervisors which specialist to consult
3. Run specialist in parallel and collect their answers
"""
