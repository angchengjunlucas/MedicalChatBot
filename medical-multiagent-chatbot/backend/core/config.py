from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from os import getenv


@dataclass(frozen=True)
class Settings:
    llm_api_base: str
    llm_api_key: str
    default_model: str
    supervisor_model: str
    cardiology_model: str
    geriatrics_model: str
    mental_model: str
    pubmed_email: str
    pubmed_tool_name: str
    vector_db_type: str
    chroma_path: str
    qdrant_url: str
    embedding_model: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        llm_api_base=getenv("LLM_API_BASE", ""),
        llm_api_key=getenv("LLM_API_KEY", ""),
        default_model=getenv("DEFAULT_MODEL", "gpt-4.1-mini"),
        supervisor_model=getenv("SUPERVISOR_MODEL", "gpt-4.1-mini"),
        cardiology_model=getenv("CARDIOLOGY_MODEL", "gpt-4.1-mini"),
        geriatrics_model=getenv("GERIATRICS_MODEL", "gpt-4.1-mini"),
        mental_model=getenv("MENTAL_MODEL", "gpt-4.1-mini"),
        pubmed_email=getenv("PUBMED_EMAIL", ""),
        pubmed_tool_name=getenv("PUBMED_TOOL_NAME", ""),
        vector_db_type=getenv("VECTOR_DB_TYPE", "chroma"),
        chroma_path=getenv("CHROMA_PATH", "./data/chroma"),
        qdrant_url=getenv("QDRANT_URL", ""),
        embedding_model=getenv("EMBEDDING_MODEL", ""),
    )
