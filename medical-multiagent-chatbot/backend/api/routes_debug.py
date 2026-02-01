from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.core.config import get_settings
from backend.storage.db import init_db
from backend.storage.repo import get_trace

router = APIRouter()


@router.get("/debug/trace/{trace_id}")
async def get_trace_by_id(trace_id: str):
    settings = get_settings()
    init_db(settings.log_db_url)
    record = get_trace(settings.log_db_url, trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return record.__dict__
