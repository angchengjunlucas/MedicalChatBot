import pytest

from backend.agents.cardiology import CardiologyAgent
from backend.agents.geriatrics import GeriatricsAgent
from backend.agents.mental_health import MentalHealthAgent
from backend.agents.supervisor import SupervisorAgent
from backend.llm.base import LLMResponse, LLMUsage
from backend.retrieval.context import ContextBundle


class FakeRouter:
    def __init__(self, content: str) -> None:
        self._content = content

    async def generate(self, role, messages, temperature, max_tokens):
        return LLMResponse(
            content=self._content,
            model="fake",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            raw=None,
        )


@pytest.mark.asyncio
async def test_supervisor_parsing() -> None:
    router = FakeRouter('{"specialists":["cardiology"],"rationale":"x","instructions":{"cardiology":"y"}}')
    agent = SupervisorAgent(router)  # type: ignore[arg-type]
    bundle = ContextBundle(query="q", rewritten_query=None, kb_passages=[], pubmed_refs=[], notes={})
    decision = await agent.run("q", bundle)
    assert decision.specialists == ["cardiology"]


@pytest.mark.asyncio
async def test_specialist_parsing() -> None:
    payload = (
        '{"summary":"s","key_points":["k"],"red_flags":["r"],'
        '"uncertainties":["u"],"evidence_links":["E"],"draft_response_text":"d"}'
    )
    bundle = ContextBundle(query="q", rewritten_query=None, kb_passages=[], pubmed_refs=[], notes={})

    card = CardiologyAgent(FakeRouter(payload))  # type: ignore[arg-type]
    ger = GeriatricsAgent(FakeRouter(payload))  # type: ignore[arg-type]
    men = MentalHealthAgent(FakeRouter(payload))  # type: ignore[arg-type]

    for agent in (card, ger, men):
        out = await agent.run("q", bundle)
        assert out.summary == "s"

# this is to check if my agents correctly parse JSON output from the LLM