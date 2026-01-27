import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.main import create_app


@pytest.mark.asyncio
async def test_kb_test_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    app: FastAPI = create_app()

    class FakeSettings:
        vector_db_type = "chroma"
        chroma_path = "/tmp/chroma"
        llm_api_base = "http://fake"
        llm_api_key = "fake"
        embedding_model = "fake-embed"

    class FakeEmbedder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def embed(self, texts):
            return [[0.1, 0.2, 0.3]]

    class FakeStore:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def query(self, query_embedding, top_k=5, where=None):
            from backend.retrieval.vectorstore import VectorRecord

            return [
                VectorRecord(
                    doc_id="doc1",
                    chunk_id="doc1:0",
                    text="hello",
                    metadata={"doc_id": "doc1", "chunk_id": "doc1:0"},
                    score=0.9,
                )
            ]

    monkeypatch.setattr("backend.api.routes_kb.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("backend.api.routes_kb.OpenAIEmbeddingsProvider", FakeEmbedder)
    monkeypatch.setattr("backend.api.routes_kb.ChromaVectorStore", FakeStore)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/kb_test", params={"q": "heart failure", "top_k": 1})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
