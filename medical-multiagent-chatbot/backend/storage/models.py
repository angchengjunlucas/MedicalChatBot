from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    user_id: str
    query: str
    created_at: str
    response_text: str
    response_detail: str | None
    context_meta: dict[str, Any]
    agent_outputs: dict[str, Any]
    safety_flags: list[str]
    token_usage: dict[str, Any]

#this basically is the log entry of one chat request, which will enable me to audit the sysmte and measure it in the future
