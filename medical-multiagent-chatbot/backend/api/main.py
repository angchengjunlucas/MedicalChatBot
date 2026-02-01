from fastapi import FastAPI

from backend.api.routes_chat import router as chat_router
from backend.api.routes_debug import router as debug_router
from backend.api.routes_kb import router as kb_router
from backend.api.routes_pubmed import router as pubmed_router
from backend.api.routes_rag import router as rag_router


def create_app() -> FastAPI:
    app = FastAPI(title="Medical Multi-Agent Chatbot", version="0.0.1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(chat_router)
    app.include_router(debug_router)
    app.include_router(kb_router)
    app.include_router(pubmed_router)
    app.include_router(rag_router)

    return app


app = create_app()

#this is for building an API server, checking if my backend is runnign and reachable
