"""LLM-assisted rule drafting (/policy-rules/draft): contract tests.

Pins: admin-only gate, draft-only semantics (the runtime rule set is never
touched by this endpoint), deterministic validation of the LLM output, and
sample self-testing. The LLM call itself is mocked — no network in tests.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from api.deps import require_system_settings
from core.auth.backend import UserContext, get_current_user
from core.db.engine import Base, get_db
from safeharness.policies.rules_loader import clear_runtime_rules, runtime_rules

_VALID_RULE = {
    "name": "block_x_marker",
    "pattern": "x_marker_xyz",
    "severity": "high",
    "category": "demo",
    "reason": "测试规则",
    "action": "deny",
}


@pytest.fixture
def client():
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

    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", user_center_id="c", username="alice",
    )
    app.dependency_overrides[require_system_settings] = lambda: "alice"
    app.dependency_overrides[get_db] = _override_db
    clear_runtime_rules()
    c = TestClient(app)
    try:
        yield c
    finally:
        clear_runtime_rules()
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(require_system_settings, None)
        app.dependency_overrides.pop(get_db, None)


def _mock_llm(monkeypatch, rules=None, exc=None, model="test-model"):
    from api.routes.v1 import safeharness as routes

    async def fake(description, samples):
        if exc is not None:
            raise exc
        return rules, model

    monkeypatch.setattr(routes, "_draft_rules_via_llm", fake)


def test_draft_requires_admin(client, monkeypatch):
    # Drop the admin override: the real dependency must reject.
    from api.app import app

    app.dependency_overrides.pop(require_system_settings, None)
    try:
        res = client.post(
            "/v1/safeharness/policy-rules/draft",
            json={"description": "拦截泄露系统提示词的输入"},
        )
        assert res.status_code in (401, 403)
    finally:
        app.dependency_overrides[require_system_settings] = lambda: "alice"


def test_draft_valid_rules_returned_and_not_applied(client, monkeypatch):
    _mock_llm(monkeypatch, rules=[_VALID_RULE])
    res = client.post(
        "/v1/safeharness/policy-rules/draft",
        json={"description": "拦测试标记", "sample_texts": ["see x_marker_xyz here"]},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["validation_errors"] == []
    assert data["draft"][0]["name"] == "block_x_marker"
    assert data["model"] == "test-model"
    assert "未应用" in data["notice"]
    # Draft-only: the runtime rule set must be untouched.
    assert runtime_rules() == ()
    # Sample self-test ran against the draft.
    assert data["sample_results"] == [
        {"text": "see x_marker_xyz here", "hit": True, "matched_rule": "block_x_marker"}
    ]


def test_draft_llm_failure_maps_to_502(client, monkeypatch):
    _mock_llm(monkeypatch, exc=ValueError("模型响应中找不到 JSON 对象"))
    res = client.post(
        "/v1/safeharness/policy-rules/draft",
        json={"description": "随便描述"},
    )
    assert res.status_code == 502
    body = res.json()
    assert body["data"]["error"]["code"] == "safeharness_rule_draft_failed"
    assert "起草失败" in body["message"]
    assert runtime_rules() == ()


def test_draft_invalid_rules_reported_not_applied(client, monkeypatch):
    duplicated = [
        {**_VALID_RULE, "name": "dup", "pattern": "a"},
        {**_VALID_RULE, "name": "dup", "pattern": "b"},
    ]
    _mock_llm(monkeypatch, rules=duplicated)
    res = client.post(
        "/v1/safeharness/policy-rules/draft",
        json={"description": "重复名草稿"},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert any("duplicate" in e for e in data["validation_errors"])
    assert data["sample_results"] == []  # samples skipped on invalid draft
    assert runtime_rules() == ()


def test_draft_allow_action_rejected(client, monkeypatch):
    allow_rule = {**_VALID_RULE, "name": "allow_all", "pattern": ".*", "action": "allow"}
    _mock_llm(monkeypatch, rules=[allow_rule])
    res = client.post(
        "/v1/safeharness/policy-rules/draft",
        json={"description": "试图放行"},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert any("action" in e for e in data["validation_errors"])
    assert runtime_rules() == ()


def test_draft_sample_results_report_misses(client, monkeypatch):
    _mock_llm(monkeypatch, rules=[_VALID_RULE])
    res = client.post(
        "/v1/safeharness/policy-rules/draft",
        json={
            "description": "拦测试标记",
            "sample_texts": ["clean text", "contains x_marker_xyz inside"],
        },
    )
    data = res.json()["data"]
    hits = {r["text"]: r["hit"] for r in data["sample_results"]}
    assert hits == {"clean text": False, "contains x_marker_xyz inside": True}
