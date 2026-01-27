import pytest

from backend.orchestration.runner import Orchestrator
from backend.retrieval.context import ContextBundle
from backend.retrieval.pipeline import RAGPipeline
from backend.retrieval.vectorstore import VectorRecord


@pytest.mark.asyncio
async def test_orchestrator_runs_specialists() -> None:
    class FakeRAG(RAGPipeline):
        async def retrieve(self, query: str, top_k: int = 5, pubmed_k: int = 5):
            return ContextBundle(query=query, rewritten_query=None, kb_passages=[], pubmed_refs=[], notes={})

    class FakeRouter:
        async def generate(self, role, messages, temperature, max_tokens):
            if role == "supervisor":
                return type("Resp", (), {"content": '{"specialists":["cardiology"]}'})
            return type("Resp", (), {"content": '{"summary":"s","key_points":[],"red_flags":[],"uncertainties":[],"evidence_links":[],"draft_response_text":"d"}'})

    orch = Orchestrator(rag=FakeRAG(None, None), router=FakeRouter())  # type: ignore[arg-type]
    result = await orch.run("q")

    assert "cardiology" in result.outputs

# checks if the orchestrators call the specialist agents that the supervisor selects