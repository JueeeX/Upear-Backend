"""Contracts for the user-facing test and configuration center."""

import asyncio

from api.routes.v1.control_center import (
    SCENARIOS,
    TestRunRequest as ControlCenterTestRunRequest,
    _diff,
    _next_version,
    _run_scenarios,
    _validate_payload,
    create_test_run,
)


def _card(**overrides):
    capability = {
        "capability_id": "alarm.read",
        "description": "Read alarm summary",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "required_capabilities": [],
        "risk_level": "low",
        "side_effects": [],
        "reversibility": "read_only",
        "idempotent": True,
        "timeout_seconds": 30,
        "max_retries": 1,
        "outcomes": ["success"],
    }
    value = {
        "schema_version": "agent-card.v1",
        "agent_id": "alarm-agent",
        "runtime": "langgraph",
        "capabilities": [capability],
    }
    value.update(overrides)
    return value


def test_config_validation_normalizes_agent_card_and_rejects_secrets():
    normalized, validation = _validate_payload({"agent_card": _card(), "planning": {"mode": "fsm"}})
    assert validation == {"valid": True, "reason_code": "config_valid"}
    assert normalized["agent_card"]["schema_version"] == "agent-card.v1"

    normalized, validation = _validate_payload({"agent_card": _card(api_key="secret")})
    assert normalized is None
    assert validation["reason_code"] == "config_sensitive_field"


def test_planning_schema_rejects_unknown_fields():
    normalized, validation = _validate_payload({"planning": {"unsafe_switch": True}})
    assert normalized is None
    assert validation["reason_code"] == "planning_unknown_field"


def test_config_diff_is_path_addressable():
    changes = _diff({"agent_card": {"runtime": "a"}}, {"agent_card": {"runtime": "b", "x": 1}})
    assert {item["path"] for item in changes} == {"agent_card.runtime", "agent_card.x"}


def test_config_version_increment_handles_semver_like_values():
    assert _next_version("V1.0") == "V2.0"
    assert _next_version("V9.2") == "V10.0"


def test_test_center_scenarios_are_fixed_and_sanitized():
    assert {item[0] for item in SCENARIOS} == {
        "input-injection", "nested-tool-result", "restored-tool-call", "restored-history"
    }
    result = asyncio.run(_run_scenarios("input-injection"))
    assert result[0]["passed"] is True
    assert set(result[0]) == {"name", "blocked_stage", "passed", "evidence"}


def test_create_test_run_persists_result(monkeypatch):
    class FakeDb:
        def __init__(self):
            self.items = []

        def add(self, item):
            self.items.append(item)

        def commit(self):
            return None

        def get(self, *_args):
            return None

    class User:
        user_id = "user-1"
        user_center_id = "tenant-1"

    async def fake_run(_scenario_id):
        return [{"name": "input-injection", "blocked_stage": "input_guard", "passed": True, "evidence": {}}]

    monkeypatch.setattr("api.routes.v1.control_center._run_scenarios", fake_run)
    monkeypatch.setattr("api.routes.v1.control_center._tenant", lambda *_args: "tenant-1")
    db = FakeDb()
    result = asyncio.run(create_test_run(ControlCenterTestRunRequest(scenario_id="all"), User(), db))
    assert result["data"]["status"] == "pass"
    assert db.items[0].run_id.startswith("test_")
