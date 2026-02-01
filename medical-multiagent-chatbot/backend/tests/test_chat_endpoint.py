import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.main import create_app


@pytest.mark.asyncio
async def test_chat_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    app: FastAPI = create_app()

    class FakeSettings:
        vector_db_type = "chroma"
        chroma_path = "/tmp/chroma"
        llm_api_base = "http://fake"
        llm_api_key = "fake"
        local_base_model = ""
        local_supervisor_adapter = ""
        local_cardiology_adapter = ""
        local_geriatrics_adapter = ""
        local_mental_adapter = ""
        embedding_model = "fake-embed"
        pubmed_email = "test@example.com"
        pubmed_tool_name = "test-tool"
        log_db_url = "sqlite:///./data/test.db"
        default_model = "fake"
        supervisor_model = "fake"
        cardiology_model = "fake"
        geriatrics_model = "fake"
        mental_model = "fake"

    class FakeOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def run(self, query: str):
            from backend.agents.base import AgentOutput

            return type(
                "Result",
                (),
                {
                    "final_text": "d",
                    "full_text": "full",
                    "context_meta": {"rewritten_query": None, "kb_count": 0, "pubmed_count": 0},
                    "outputs": {
                        "cardiology": AgentOutput(
                            summary="s",
                            key_points=[],
                            red_flags=[],
                            uncertainties=[],
                            evidence_links=[],
                            draft_response_text="d",
                        )
                    }
                },
            )()

    monkeypatch.setattr("backend.api.routes_chat.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("backend.api.routes_chat.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("backend.api.routes_chat.ChromaVectorStore", lambda *a, **k: object())
    monkeypatch.setattr("backend.api.routes_chat.OpenAIEmbeddingsProvider", lambda *a, **k: object())
    monkeypatch.setattr("backend.api.routes_chat.KBRetrievalPipeline", lambda *a, **k: object())
    monkeypatch.setattr("backend.api.routes_chat.PubMedClient", lambda *a, **k: object())
    monkeypatch.setattr("backend.api.routes_chat.RAGPipeline", lambda *a, **k: object())
    monkeypatch.setattr("backend.api.routes_chat.init_db", lambda *a, **k: None)
    monkeypatch.setattr("backend.api.routes_chat.save_trace", lambda *a, **k: None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"user_id": "u1", "query": "q", "history": [], "user_metadata": {}},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["response_text"] == "d"
    assert payload["response_detail"] == "full"

#ensure that a final response text is actually given back without real RAG and the agents
