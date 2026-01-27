import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.main import create_app
from backend.llm.base import LLMResponse, LLMUsage


@pytest.mark.anyio
async def test_llm_test_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    app: FastAPI = create_app()

    class FakeProvider:
        async def generate(self, model_id, messages, temperature, max_tokens):
            return LLMResponse(
                content="hello",
                model=model_id,
                usage=LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                raw=None,
            )

    class FakeRouter:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, role, messages, temperature, max_tokens):
            return await FakeProvider().generate("fake-model", messages, temperature, max_tokens)

    class FakeSettings:
        llm_api_base = "http://fake"
        llm_api_key = "fake"
        default_model = "fake-model"
        supervisor_model = "fake-model"
        cardiology_model = "fake-model"
        geriatrics_model = "fake-model"
        mental_model = "fake-model"

    monkeypatch.setattr("backend.api.routes_chat.ModelRouter", FakeRouter)
    monkeypatch.setattr("backend.api.routes_chat.get_settings", lambda: FakeSettings())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/llm_test",
            json={
                "role": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.2,
                "max_tokens": 5,
            },
        )

    assert resp.status_code == 200
    assert resp.json()["content"] == "hello"
