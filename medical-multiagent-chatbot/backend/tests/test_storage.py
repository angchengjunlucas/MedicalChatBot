from backend.storage.db import init_db
from backend.storage.models import TraceRecord
from backend.storage.repo import get_trace, now_utc_iso, save_trace


def test_save_and_get_trace(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path}/test.db"
    init_db(db_url)

    record = TraceRecord(
        trace_id="t1",
        user_id="u1",
        query="q",
        created_at=now_utc_iso(),
        response_text="r",
        response_detail="full",
        context_meta={"a": 1},
        agent_outputs={"cardiology": {"summary": "s"}},
        safety_flags=["f"],
        token_usage={"prompt": 1},
    )
    save_trace(db_url, record)
    loaded = get_trace(db_url, "t1")
    assert loaded is not None
    assert loaded.trace_id == "t1"
