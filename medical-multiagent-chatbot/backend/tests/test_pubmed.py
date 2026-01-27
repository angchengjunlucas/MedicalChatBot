import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.main import create_app


@pytest.mark.asyncio
async def test_pubmed_test_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    app: FastAPI = create_app()

    class FakeSettings:
        pubmed_email = "test@example.com"
        pubmed_tool_name = "test-tool"

    class FakePubMedClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def search_and_summary(self, query: str, retmax: int = 5):
            return [
                {
                    "pmid": "1",
                    "title": "Test",
                    "year": 2020,
                    "journal": "Journal",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                    "authors": ["A"],
                    "abstract": None,
                }
            ]

    monkeypatch.setattr("backend.api.routes_pubmed.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("backend.api.routes_pubmed.PubMedClient", FakePubMedClient)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/pubmed_test", params={"q": "asthma", "retmax": 1})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
