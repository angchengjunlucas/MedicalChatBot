from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from backend.storage.db import _sqlite_path
from backend.storage.models import TraceRecord


def save_trace(db_url: str, record: TraceRecord) -> None:
    path = _sqlite_path(db_url)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO traces (
                trace_id, user_id, query, created_at, response_text,
                context_meta, agent_outputs, safety_flags, token_usage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.trace_id,
                record.user_id,
                record.query,
                record.created_at,
                record.response_text,
                json.dumps(record.context_meta),
                json.dumps(record.agent_outputs),
                json.dumps(record.safety_flags),
                json.dumps(record.token_usage),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_trace(db_url: str, trace_id: str) -> TraceRecord | None:
    path = _sqlite_path(db_url)
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        row = cur.fetchone()
        if not row:
            return None
        return TraceRecord(
            trace_id=row[0],
            user_id=row[1],
            query=row[2],
            created_at=row[3],
            response_text=row[4],
            context_meta=json.loads(row[5]),
            agent_outputs=json.loads(row[6]),
            safety_flags=json.loads(row[7]),
            token_usage=json.loads(row[8]),
        )
    finally:
        conn.close()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
