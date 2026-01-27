from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DBConfig:
    url: str


def _sqlite_path(db_url: str) -> Path:
    if not db_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite URLs are supported for dev")
    return Path(db_url.replace("sqlite:///", "", 1))


def init_db(db_url: str) -> None:
    path = _sqlite_path(db_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                user_id TEXT,
                query TEXT,
                created_at TEXT,
                response_text TEXT,
                context_meta TEXT,
                agent_outputs TEXT,
                safety_flags TEXT,
                token_usage TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
