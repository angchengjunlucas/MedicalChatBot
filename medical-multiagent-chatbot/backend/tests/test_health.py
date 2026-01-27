import httpx
import pytest

from backend.api.main import create_app


@pytest.mark.anyio
async def test_health_endpoint() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

#this is like a test for the FASTApi