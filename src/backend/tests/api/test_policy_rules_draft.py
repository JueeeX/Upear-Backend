"""LLM-assisted rule drafting (/policy-rules/draft): contract tests.

Pins: admin-only gate, draft-only semantics (the runtime rule set is never
touched by this endpoint), deterministic validation of the LLM output, and
sample self-testing. The LLM call itself is mocked — no network in tests.
"""

import json

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


def _mock_llm(monkeypatch, rules=None, exc=None, model="test-model", chains=None):
    from api.routes.v1 import safeharness as routes

    async def fake(description, samples):
        if exc is not None:
            raise exc
        return rules, chains or [], model

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


# ── P1.3: chain drafting + attack-corpus regression preview ───────────────

_VALID_CHAIN = {
    "name": "chain_read_then_send",
    "first": "query_db.*",
    "then": "http_post.*",
    "max_gap": 2,
    "severity": "critical",
    "reason": "先查库后外发",
}


def test_draft_with_chains_validated_not_applied(client, monkeypatch):
    _mock_llm(monkeypatch, rules=[_VALID_RULE], chains=[_VALID_CHAIN])
    res = client.post(
        "/v1/safeharness/policy-rules/draft",
        json={"description": "先查库再外发的场景"},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["validation_errors"] == []
    assert data["chain_draft"][0]["name"] == "chain_read_then_send"
    assert runtime_rules() == ()
    from safeharness.policies.rules_loader import runtime_chains

    assert runtime_chains() == ()  # chains are draft-only too


def test_draft_chain_errors_reported_rules_kept(client, monkeypatch):
    broken = [{"name": "chain_broken", "first": "a.*", "then": "", "max_gap": 1, "severity": "high"}]
    _mock_llm(monkeypatch, rules=[_VALID_RULE], chains=broken)
    res = client.post(
        "/v1/safeharness/policy-rules/draft",
        json={"description": "链缺端点条件"},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert any("then" in e for e in data["validation_errors"])
    assert data["draft"]  # the valid rule half survives


def test_draft_without_chains_keeps_legacy_shape(client, monkeypatch):
    _mock_llm(monkeypatch, rules=[_VALID_RULE])
    res = client.post("/v1/safeharness/policy-rules/draft", json={"description": "无链场景"})
    data = res.json()["data"]
    assert data["chain_draft"] == []
    assert set(data) >= {"draft", "chain_draft", "validation_errors", "sample_results", "regression", "model", "notice"}


def test_draft_regression_preview_numbers(client, monkeypatch, tmp_path):
    # A tiny corpus: one attack the draft catches, one it does not.
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "demo.json").write_text(
        json.dumps(
            {
                "name": "demo-suite",
                "description": "t",
                "attacks": [
                    {"id": "a1", "severity": "high", "text": "see x_marker_xyz here"},
                    {"id": "a2", "severity": "high", "text": "totally innocent text"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SAFEHARNESS_ATTACK_SUITES_DIR", str(suites))
    _mock_llm(monkeypatch, rules=[_VALID_RULE])
    res = client.post("/v1/safeharness/policy-rules/draft", json={"description": "标记拦截"})
    data = res.json()["data"]
    regression = data["regression"]
    assert regression is not None
    assert regression["total"] == 2
    assert regression["draft_caught"] == 1
    assert regression["escapes"] == ["demo-suite/a2"]
    assert regression["per_rule"] == [{"name": "block_x_marker", "fired": 1}]


def test_draft_regression_absent_when_suites_missing(client, monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_ATTACK_SUITES_DIR", "Z:/definitely/missing/suites")
    _mock_llm(monkeypatch, rules=[_VALID_RULE])
    res = client.post("/v1/safeharness/policy-rules/draft", json={"description": "无套件"})
    data = res.json()["data"]
    assert data["regression"] is None
    assert "攻击集回归不可用" in data["notice"]


def test_draft_regression_absent_when_validation_failed(client, monkeypatch):
    duplicated = [
        {**_VALID_RULE, "name": "dup", "pattern": "a"},
        {**_VALID_RULE, "name": "dup", "pattern": "b"},
    ]
    _mock_llm(monkeypatch, rules=duplicated)
    res = client.post("/v1/safeharness/policy-rules/draft", json={"description": "校验失败"})
    data = res.json()["data"]
    assert data["validation_errors"]
    assert data["regression"] is None
