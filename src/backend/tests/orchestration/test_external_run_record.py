"""External-agent /stream branch run records (P3.3).

Pins: ``start_external_run`` creates a ``chat_runs`` row and drives it to a
terminal state from the background turn — completed with the response and
frame offsets, failed with the error recorded, user-cancel preserved. The
Redis stream writes are fail-open here (no live Redis in unit tests), which
the run machinery shares with the workflow path by construction.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.engine import Base
from core.db.models import ChatRun
from orchestration import chat_run_executor


class _FakeRedis:
    """Records xadd'd frames so tests can pin the frame contract."""

    def __init__(self):
        self.events: list[dict] = []

    async def xadd(self, key, fields, maxlen=None, approximate=True):
        self.events.append(json.loads(fields["data"]))

    async def expire(self, key, ttl):
        pass


@pytest.fixture
def run_db(monkeypatch):
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    monkeypatch.setattr(chat_run_executor, "SessionLocal", Session)
    fake = _FakeRedis()
    monkeypatch.setattr(chat_run_executor, "get_redis", lambda: fake)
    return Session, fake


def _events(fake) -> list[str]:
    return [e.get("type") for e in fake.events if e.get("type") != "__terminal__"]


def _raw_events(fake) -> list[dict]:
    return list(fake.events)


def _get_run(Session, run_id: str) -> ChatRun:
    s = Session()
    try:
        return s.query(ChatRun).filter(ChatRun.run_id == run_id).one()
    finally:
        s.close()


async def _await_terminal(Session, run_id: str, timeout: float = 5.0) -> ChatRun:
    async def _wait():
        while True:
            row = _get_run(Session, run_id)
            if row.status not in ("pending", "running"):
                return row
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(_wait(), timeout)


def test_external_run_completes_with_response(run_db):
    Session, fake = run_db

    async def _scenario():
        run = await chat_run_executor.start_external_run(
            chat_id="c1",
            user_id="u1",
            request_payload={"message": "hi"},
            run_turn=lambda: asyncio.sleep(0, result=SimpleNamespace(response="done")),
        )
        row = await _await_terminal(Session, run.run_id)
        return run, row

    run, row = asyncio.run(_scenario())
    assert row.status == "completed"
    assert row.chat_id == "c1" and row.user_id == "u1"
    assert row.last_event_offset == 2
    assert not row.error_message
    assert _events(fake) == ["text", "end"]
    text_frame = next(e for e in _raw_events(fake) if e.get("type") == "text")
    assert text_frame["content"] == "done"
    assert text_frame["chat_id"] == "c1"
    # the terminal marker closes the stream for the follower
    assert _raw_events(fake)[-1]["type"] == "__terminal__"


def test_external_run_records_failure(run_db):
    Session, fake = run_db

    async def _boom():
        raise RuntimeError("model exploded")

    async def _scenario():
        run = await chat_run_executor.start_external_run(
            chat_id="c2",
            user_id="u1",
            request_payload={"message": "hi"},
            run_turn=_boom,
        )
        row = await _await_terminal(Session, run.run_id)
        return row

    row = asyncio.run(_scenario())
    assert row.status == "failed"
    assert "model exploded" in (row.error_message or "")
    # error frame + terminal marker were emitted (the marker is filtered out of _events)
    assert _events(fake) == ["error"]
    assert _raw_events(fake)[-1]["type"] == "__terminal__"


def test_external_run_preserves_cancel(run_db):
    Session, fake = run_db

    async def _cancelled():
        raise asyncio.CancelledError()

    async def _scenario():
        run = await chat_run_executor.start_external_run(
            chat_id="c3",
            user_id="u1",
            request_payload={"message": "hi"},
            run_turn=_cancelled,
        )
        row = await _await_terminal(Session, run.run_id)
        return row

    row = asyncio.run(_scenario())
    assert row.status == "cancelled"
