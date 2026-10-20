"""M10 custom external-agent catalog — data-driven registration, hot-effective.

The builtin CATALOG stays code-side; operator-registered cards live in
system_configs (``external_agent_custom.<agent_id>``) and merge at read time.
framework is whitelist-bound to RUNNERS at write time and re-verified at read
time; unregistered/removed agents keep raising the historic KeyError/400.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.auth.backend import UserContext, get_current_user
from core.db.engine import Base, get_db
from core.db.models import SystemConfig
from core.external_agents.custom_registry import (
    CustomAgentError,
    delete_custom,
    get_custom,
    list_custom,
    lookup_agent,
    merged_catalog,
    register_custom,
    update_custom,
)
from core.external_agents.registry import CATALOG, run_external_agent
from core.external_agents.runners import RUNNERS, RunnerResult

AGENT_ID = "custom-helper"


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


@pytest.fixture
def client(engine, monkeypatch):
    Session = sessionmaker(bind=engine)
    from api.app import app
    from api.deps import require_system_settings

    def _override_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", user_center_id="c", username="alice"
    )
    app.dependency_overrides[require_system_settings] = lambda: "alice"
    app.dependency_overrides[get_db] = _override_db
    yield TestClient(app)
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(require_system_settings, None)
    app.dependency_overrides.pop(get_db, None)


def _payload(agent_id: str = AGENT_ID, **over) -> dict[str, Any]:
    body = {
        "agent_id": agent_id,
        "name": "自定义运维助手",
        "framework": "smolagents",
        "project": "internal/ops",
        "description": "运营注册的自定义智能体（测试）",
    }
    body.update(over)
    return body


# ---------- unit: custom_registry ----------


def test_register_and_merged_catalog_roundtrip(session):
    record = register_custom(session, **_payload(), actor="op")
    assert record["agent_id"] == AGENT_ID and record["enabled"] is True
    assert AGENT_ID in list_custom(session)
    merged = merged_catalog(session)
    assert AGENT_ID in merged
    assert merged[AGENT_ID].framework == "smolagents"
    assert merged[AGENT_ID].name == "自定义运维助手"
    assert lookup_agent(session, AGENT_ID) == merged[AGENT_ID]
    # builtin lookup stays untouched
    assert lookup_agent(session, "ext-smolagents") is CATALOG["ext-smolagents"]


def test_register_rejects_bad_agent_id(session):
    for bad in ("Bad-ID", "ab", "x" * 64, "ext-smolagents", "has_underscore"):
        with pytest.raises(CustomAgentError):
            register_custom(session, **_payload(agent_id=bad), actor="op")


def test_register_rejects_unknown_framework(session):
    with pytest.raises(CustomAgentError):
        register_custom(session, **_payload(framework="not-a-framework"), actor="op")


def test_register_rejects_duplicate(session):
    register_custom(session, **_payload(), actor="op")
    with pytest.raises(CustomAgentError):
        register_custom(session, **_payload(), actor="op")


def test_update_fields_and_disable(session):
    register_custom(session, **_payload(), actor="op")
    record = update_custom(
        session,
        AGENT_ID,
        {"name": "改名后的助手", "enabled": False},
        actor="op2",
    )
    assert record["name"] == "改名后的助手"
    assert record["updated_by"] == "op2"
    # disabled: kept in records, dropped from the effective catalog
    assert AGENT_ID in list_custom(session)
    assert AGENT_ID not in merged_catalog(session)
    assert lookup_agent(session, AGENT_ID) is None
    update_custom(session, AGENT_ID, {"enabled": True}, actor="op")
    assert AGENT_ID in merged_catalog(session)


def test_update_unknown_raises(session):
    with pytest.raises(CustomAgentError):
        update_custom(session, "no-such-agent", {"name": "x"}, actor="op")


def test_delete_removes_record_and_residue(session):
    register_custom(session, **_payload(), actor="op")
    session.add(
        SystemConfig(
            config_key=f"external_agent_card.{AGENT_ID}",
            config_value=json.dumps({"name": "x"}),
            display_name="card",
            group_key="external_agents",
            is_secret=False,
        )
    )
    session.add(
        SystemConfig(
            config_key=f"external_agent_model.agent.{AGENT_ID}",
            config_value=json.dumps({"provider_id": "p1"}),
            display_name="model",
            group_key="external_agents",
            is_secret=False,
        )
    )
    session.commit()
    delete_custom(session, AGENT_ID)
    assert get_custom(session, AGENT_ID) is None
    for residue in (
        f"external_agent_card.{AGENT_ID}",
        f"external_agent_model.agent.{AGENT_ID}",
    ):
        assert session.get(SystemConfig, residue) is None


def test_delete_builtin_rejected(session):
    with pytest.raises(CustomAgentError):
        delete_custom(session, "ext-autogen")


def test_broken_rows_are_skipped(session):
    session.add(
        SystemConfig(
            config_key="external_agent_custom.broken-one",
            config_value="{not json",
            display_name="broken",
            group_key="external_agents",
            is_secret=False,
        )
    )
    session.commit()
    assert "broken-one" not in merged_catalog(session)
    assert "broken-one" not in list_custom(session)


# ---------- api: endpoints ----------


def test_register_endpoint_then_list_and_card(client):
    res = client.post("/v1/external-agents", json=_payload())
    assert res.status_code == 200
    assert res.json()["data"]["agent_id"] == AGENT_ID

    listing = client.get("/v1/external-agents").json()["data"]
    by_id = {a["agent_id"]: a for a in listing["agents"]}
    assert by_id[AGENT_ID]["custom"] is True
    assert by_id["ext-smolagents"]["custom"] is False
    assert AGENT_ID in listing["cards"]

    card = client.get(f"/v1/external-agents/{AGENT_ID}/card").json()["data"]
    assert card["name"] == "自定义运维助手"


def test_register_requires_admin(client):
    from api.deps import require_system_settings

    client.app.dependency_overrides.pop(require_system_settings, None)
    try:
        res = client.post("/v1/external-agents", json=_payload())
        assert res.status_code in (401, 403)
    finally:
        client.app.dependency_overrides[require_system_settings] = lambda: "alice"


def test_register_validation_returns_400(client):
    res = client.post("/v1/external-agents", json=_payload(framework="nope"))
    assert res.status_code == 400
    assert "framework" in res.json()["message"]


def test_update_endpoint_toggle(client):
    client.post("/v1/external-agents", json=_payload())
    res = client.put(f"/v1/external-agents/{AGENT_ID}", json={"enabled": False})
    assert res.status_code == 200
    listing = client.get("/v1/external-agents").json()["data"]
    assert AGENT_ID not in {a["agent_id"] for a in listing["agents"]}
    client.put(f"/v1/external-agents/{AGENT_ID}", json={"enabled": True})
    listing = client.get("/v1/external-agents").json()["data"]
    assert AGENT_ID in {a["agent_id"] for a in listing["agents"]}


def test_delete_endpoint_then_run_400(client):
    client.post("/v1/external-agents", json=_payload())
    assert client.delete(f"/v1/external-agents/{AGENT_ID}").status_code == 200
    res = client.post(f"/v1/external-agents/{AGENT_ID}/run", json={"message": "你好"})
    assert res.status_code == 400
    assert client.delete(f"/v1/external-agents/{AGENT_ID}").status_code == 400


def test_delete_builtin_400(client):
    res = client.delete("/v1/external-agents/ext-smolagents")
    assert res.status_code == 400


def test_model_settings_scope_accepts_custom(client, monkeypatch):
    import core.external_agents.model_settings as ms

    monkeypatch.setattr(ms, "_validate_provider", lambda db, provider_id: None)
    client.post("/v1/external-agents", json=_payload())
    res = client.put(
        "/v1/external-agents/model-settings",
        json={"scope": AGENT_ID, "provider_id": "p1"},
    )
    assert res.status_code == 200
    assert res.json()["data"]["agents"][AGENT_ID] == "p1"


def test_run_custom_agent_end_to_end(client, monkeypatch, engine):
    """Registered card runs through the same guarded bridge (fake runner)."""
    client.post("/v1/external-agents", json=_payload())

    async def _fake_runner(message, config):
        return RunnerResult(
            reply="自定义智能体已执行",
            framework="smolagents",
            model="m",
            duration_ms=3,
            steps=1,
        )

    monkeypatch.setitem(RUNNERS, "smolagents", _fake_runner)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "core.external_agents.registry._model_provider_config",
        lambda agent_id: {
            "base_url": "http://example.invalid/v1",
            "api_key": "k",
            "model": "m",
            "provider_id": "main_agent",
            "source": "main_agent",
        },
    )
    monkeypatch.setattr("core.external_agents.registry._record_run", _noop)

    # run_external_agent re-opens SessionLocal for the custom lookup: bind it
    # to the shared in-memory engine so the registered row is visible.
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr("core.db.engine.SessionLocal", TestSession)

    res = client.post(f"/v1/external-agents/{AGENT_ID}/run", json={"message": "你好"})
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["reply"] == "自定义智能体已执行"
    assert data["evidence"]["guard"] == "passed"
    assert data["evidence"]["agent_id"] == AGENT_ID
