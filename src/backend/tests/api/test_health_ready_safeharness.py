"""Gated ``safeharness`` addendum on GET /ready (P3.1) + strict flip (P3.3).

Pins: absent by default (bit-identical responses), present with
audit_db/tool_gateway/protected_chat_enabled when SAFEHARNESS_HEALTH_DETAIL=1,
and purely informational by default. With SAFEHARNESS_READY_STRICT=1 a
configured-but-broken audit db or tool gateway flips the verdict to 503
(absent components and protected_chat_enabled never do), and the section is
reported even without the detail flag so the 503 reason stays visible.
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
        "SAFEHARNESS_READY_STRICT",
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


# ── Strict flip (P3.3, SAFEHARNESS_READY_STRICT) ─────────────────────────────


def test_ready_strict_flips_on_broken_configured_gateway(client, monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_READY_STRICT", "1")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_API_KEY", "k" * 40)
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    # The section is reported even without the detail flag: a 503 must show why.
    assert body["safeharness"]["tool_gateway"]["ok"] is False


def test_ready_strict_flips_on_broken_configured_audit_db(client, monkeypatch, tmp_path):
    monkeypatch.setenv("SAFEHARNESS_READY_STRICT", "1")
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(tmp_path / "audit-empty.db"))
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["safeharness"]["audit_db"]["path_set"] is True
    assert body["safeharness"]["audit_db"]["ok"] is False


def test_ready_strict_passes_with_absent_components(client, monkeypatch):
    # Absent components are deployment choices, never verdict breaches.
    monkeypatch.setenv("SAFEHARNESS_READY_STRICT", "1")
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["safeharness"]["audit_db"]["path_set"] is False
    assert body["safeharness"]["tool_gateway"]["url_configured"] is False


def test_ready_strict_healthy_gateway_stays_ready(client, monkeypatch):
    # A configured gateway that answers /health must not block readiness.
    monkeypatch.setenv("SAFEHARNESS_READY_STRICT", "1")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_API_KEY", "k" * 40)

    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    import sys
    import types

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.AsyncClient = _FakeClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["safeharness"]["tool_gateway"]["ok"] is True
