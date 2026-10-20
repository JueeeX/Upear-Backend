"""M14 tenant-isolation phase 1 — ownership fixes, unified resolution, quotas.

Contract: artifact download/preview require an authenticated owner; memory
edit/delete verify ownership (unresolvable = not owned); evolution turn
summaries are viewer-scoped; UserContext carries a resolved tenant_id;
quotas fail open (absent config = unlimited, pre-M14 behaviour bit-identical).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.auth.account_policy import account_capacity_block_reason
from core.auth.backend import UserContext, _with_tenant
from core.auth.tenancy import load_tenant_quota, tenant_of
from core.db.engine import Base
from core.db.models import ChatMessage, ChatRun, ChatSession, SystemConfig, UserShadow


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db(engine):
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _seed_user(db, user_id: str, tenant: str | None = None):
    extra = {"tenant_id": tenant} if tenant else {}
    db.add(UserShadow(user_id=user_id, user_center_id=f"c-{user_id}", username=user_id, extra_data=extra))
    db.commit()


# ── 1. unified tenant resolution ─────────────────────────────────────────


def test_with_tenant_reads_shadow_declaration(db, monkeypatch):
    _seed_user(db, "u-tenant", tenant="acme")
    from core.db.engine import SessionLocal

    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=db.bind))
    stamped = _with_tenant(UserContext(user_id="u-tenant", user_center_id="c", username="u"))
    assert stamped.tenant_id == "acme"


def test_with_tenant_defaults_when_undeclared(db, monkeypatch):
    _seed_user(db, "u-plain")
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=db.bind))
    stamped = _with_tenant(UserContext(user_id="u-plain", user_center_id="c", username="u"))
    assert stamped.tenant_id == "default"


def test_tenant_of_degrades_on_unknown_user(monkeypatch):
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=create_engine("sqlite://")))
    assert tenant_of("nobody") == "default"


# ── 2. files ownership gate ──────────────────────────────────────────────


def test_authorize_access_anonymous_is_401():
    from api.routes.files import _authorize_access

    with pytest.raises(HTTPException) as err:
        _authorize_access(
            item={"metadata": {}}, file_id="f1", user=None, db=None, denied_action="file.download.denied"
        )
    assert err.value.status_code == 401


def test_artifact_metadata_owner_mismatch_denied(db):
    from core.services.artifact_edition import can_access_artifact_metadata

    assert can_access_artifact_metadata(db, "u1", {"owner_id": "u1"})
    assert not can_access_artifact_metadata(db, "u2", {"owner_id": "u1"})


# ── 3. memories ownership gate ───────────────────────────────────────────


def test_memory_owner_unresolvable_is_not_owned(monkeypatch):
    monkeypatch.setenv("MEM0_ENABLED", "false")
    import core.memory.service as ms

    asyncio.run(ms._memory_reset_if_needed()) if hasattr(ms, "_memory_reset_if_needed") else None
    owner = asyncio.run(ms.get_memory_owner("m-1"))
    assert owner is None  # callers must treat as not-owned


def test_memory_owner_reads_mem0_user_id(monkeypatch):
    import core.memory.service as ms

    class _FakeMemory:
        def get(self, memory_id):
            if memory_id != "m-1":
                return None
            return {"id": memory_id, "memory": "text", "user_id": "u1"}

    monkeypatch.setattr(ms, "_get_memory", lambda: _FakeMemory())
    assert asyncio.run(ms.get_memory_owner("m-1")) == "u1"
    assert asyncio.run(ms.get_memory_owner("m-other")) is None


# ── 4. evolution turn viewer scoping ─────────────────────────────────────


def test_load_summary_is_viewer_scoped(db, monkeypatch):
    from core.evolution import settlement_store

    # SessionLocal is bound at settlement_store import time — patch its own namespace.
    monkeypatch.setattr(settlement_store, "SessionLocal", sessionmaker(bind=db.bind))
    db.add(ChatSession(chat_id="ch1", user_id="u1", title="t"))
    db.add(
        ChatMessage(
            message_id="m1",
            chat_id="ch1",
            role="assistant",
            content="x",
            extra_data={settlement_store.METADATA_KEY: {"state": "settled", "mechanisms": [1]}},
        )
    )
    db.commit()

    assert settlement_store.load_summary("m1", viewer_user_id="u1") is not None
    assert settlement_store.load_summary("m1", viewer_user_id="u2") is None
    assert settlement_store.load_summary("missing") is None


# ── 5. quota framework: fail-open + enforcement ──────────────────────────


def test_load_tenant_quota_absent_is_empty(db, monkeypatch):
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=db.bind))
    assert load_tenant_quota("default") == {}
    assert load_tenant_quota("") == {}


def test_load_tenant_quota_reads_config(db, monkeypatch):
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=db.bind))
    s = sessionmaker(bind=db.bind)()
    s.add(
        SystemConfig(
            config_key="tenant_quota",
            config_value=json.dumps({"acme": {"max_users": 5, "daily_chat_runs_per_user": 3}}),
            display_name="q",
            group_key="tenant_quota",
            is_secret=False,
        )
    )
    s.commit()
    quota = load_tenant_quota("acme")
    assert quota == {"max_users": 5, "daily_chat_runs_per_user": 3}
    assert load_tenant_quota("other") == {}


def test_account_capacity_blocks_default_bucket(db, monkeypatch):
    from core.auth import tenancy

    for i in range(3):
        _seed_user(db, f"plain-{i}")
    _seed_user(db, "declared", tenant="acme")  # declared tenants don't count

    monkeypatch.setattr(
        tenancy,
        "load_tenant_quota",
        lambda tenant_id: {"max_users": 3} if tenant_id == "default" else {},
    )
    reason = account_capacity_block_reason(db)
    assert reason and "3/3" in reason

    monkeypatch.setattr(
        tenancy,
        "load_tenant_quota",
        lambda tenant_id: {"max_users": 9} if tenant_id == "default" else {},
    )
    assert account_capacity_block_reason(db) is None


def test_account_capacity_fails_open(db, monkeypatch):
    from core.auth import tenancy

    monkeypatch.setattr(tenancy, "load_tenant_quota", lambda tenant_id: {})
    assert account_capacity_block_reason(db) is None


def _chat_quota_user(db, monkeypatch, *, daily, runs_today):
    from core.auth import tenancy

    user = UserContext(user_id="u-quota", user_center_id="c", username="u", tenant_id="acme")
    monkeypatch.setattr(
        tenancy,
        "load_tenant_quota",
        lambda tenant_id: {"daily_chat_runs_per_user": daily} if tenant_id == "acme" else {},
    )
    for _ in range(runs_today):
        db.add(
            ChatRun(
                run_id=f"r-{user.user_id}-{runs_today}-{datetime.now(timezone.utc).timestamp()}-{id(user)}",
                chat_id="ch",
                user_id=user.user_id,
                message_id="m",
                status="completed",
            )
        )
    db.commit()
    return user


def test_chat_run_quota_blocks_over_daily(db, monkeypatch):
    from api.routes.v1.chats import _enforce_chat_run_quota

    user = _chat_quota_user(db, monkeypatch, daily=2, runs_today=2)
    with pytest.raises(HTTPException) as err:
        _enforce_chat_run_quota(user, db)
    assert err.value.status_code == 429


def test_chat_run_quota_allows_under_daily(db, monkeypatch):
    from api.routes.v1.chats import _enforce_chat_run_quota

    user = _chat_quota_user(db, monkeypatch, daily=5, runs_today=2)
    _enforce_chat_run_quota(user, db)  # no raise


def test_chat_run_quota_fail_open_without_config(db, monkeypatch):
    from api.routes.v1.chats import _enforce_chat_run_quota

    user = _chat_quota_user(db, monkeypatch, daily=0, runs_today=99)
    _enforce_chat_run_quota(user, db)  # no raise — unlimited
