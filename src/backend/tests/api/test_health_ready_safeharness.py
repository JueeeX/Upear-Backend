"""Gated ``safeharness`` addendum on GET /ready (P3.1).

Pins: absent by default (bit-identical responses), present with
audit_db/tool_gateway/protected_chat_enabled when SAFEHARNESS_HEALTH_DETAIL=1,
and purely informational — the ready verdict never depends on it.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from core.db.engine import Base, get_db


@pytest.fixture
def client(_clean_env):
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)

    from api.app import app

    def _override_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override_db
    # Plain construction: no lifespan startup (the app's own startup would
    # bootstrap the real DATABASE_URL, which is not this test's fixture DB).
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "SAFEHARNESS_HEALTH_DETAIL",
        "SAFEHARNESS_PROTECTED_CHAT_ENABLED",
        "SAFEHARNESS_AUDIT_DB",
        "DATABASE_URL",
        "SAFEHARNESS_TOOL_GATEWAY_URL",
        "SAFEHARNESS_TOOL_GATEWAY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    # Settings freeze at import (collection time), so blank the live settings
    # object's Redis URL (empty = check skipped) instead of the compose host.
    # RedisSettings is a frozen dataclass — go through object.__setattr__ and
    # restore afterwards.
    from core.config.settings import settings

    original_url = settings.redis.url
    object.__setattr__(settings.redis, "url", "")
    yield
    object.__setattr__(settings.redis, "url", original_url)


def test_ready_default_has_no_safeharness_section(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert "safeharness" not in body


def test_ready_detail_adds_safeharness_section(client, monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_HEALTH_DETAIL", "1")
    monkeypatch.setenv("SAFEHARNESS_PROTECTED_CHAT_ENABLED", "true")
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    section = body["safeharness"]
    assert section["protected_chat_enabled"] is True
    assert section["audit_db"] == {"path_set": False, "ok": False}
    assert section["tool_gateway"] == {"url_configured": False, "ok": None}


def test_ready_detail_never_flips_verdict(client, monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_HEALTH_DETAIL", "1")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_API_KEY", "k" * 40)
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["safeharness"]["tool_gateway"]["ok"] is False
