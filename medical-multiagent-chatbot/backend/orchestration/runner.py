from __future__ import annotations

import asyncio
from dataclasses import dataclass

from backend.agents.base import AgentOutput
from backend.agents.cardiology import CardiologyAgent
from backend.agents.geriatrics import GeriatricsAgent
from backend.agents.mental_health import MentalHealthAgent
from backend.agents.supervisor import SupervisorAgent
from backend.llm.router import ModelRouter
from backend.retrieval.context import ContextBundle
from backend.retrieval.pipeline import RAGPipeline


@dataclass(frozen=True)
class OrchestrationResult:
    supervisor_rationale: str
    outputs: dict[str, AgentOutput]


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
        return OrchestrationResult(
            supervisor_rationale=decision.rationale,
            outputs=outputs,
        )

    @staticmethod
    async def _run_agent(agent, query: str, context: ContextBundle):
        output = await agent.run(query, context)
        return agent.name, output

"""
This manager runs the whole chatbot flow
It does three steps:
1. RAG
2. Ask supervisors which specialist to consult
3. Run specialist in parallel and collect their answers
"""