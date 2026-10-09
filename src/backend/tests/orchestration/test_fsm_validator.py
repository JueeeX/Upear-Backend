import json
from types import SimpleNamespace

from core.services.plan_service import PlanService
from orchestration.subagents.fsm_contracts import (
    agent_cards_model_view,
    agent_registry_hash,
    default_agent_cards,
)
from orchestration.subagents.fsm_builder import build_fsm_from_proposal
from orchestration.subagents.fsm_validator import validate_fsm


def _valid_fsm():
    return {
        "schema_version": "fsm.v1",
        "task": {
            "goal": "repair device",
            "inputs": {
                "device_id": "NE-1",
                "interface_id": "ge-0/0/1",
                "ticket_id": "INC-TEST",
                "reason": "test",
            },
            "success_criteria": ["alarm resolved"],
        },
        "initial_state": "query_before",
        "states": [
            {
                "state_id": "query_before",
                "kind": "agent",
                "agent_id": "noc-diagnosis-agent",
                "capability_id": "alarm.query",
                "input_bindings": {"device_id": {"source": "task", "key": "device_id"}},
            },
            {
                "state_id": "diagnose_before",
                "kind": "agent",
                "agent_id": "noc-diagnosis-agent",
                "capability_id": "alarm.diagnose",
                "input_bindings": {
                    "device_id": {"source": "task", "key": "device_id"},
                    "alarms": {"source": "state", "state_id": "query_before", "key": "alarms"},
                },
            },
            {"state_id": "approve_restart", "kind": "approval", "approval_for": "restart_interface"},
            {
                "state_id": "restart_interface",
                "kind": "agent",
                "agent_id": "config-change-agent",
                "capability_id": "interface.restart",
                "input_bindings": {
                    "device_id": {"source": "task", "key": "device_id"},
                    "interface": {"source": "task", "key": "interface_id"},
                    "ticket_id": {"source": "task", "key": "ticket_id"},
                    "reason": {"source": "task", "key": "reason"},
                },
            },
            {
                "state_id": "query_after",
                "kind": "agent",
                "agent_id": "noc-diagnosis-agent",
                "capability_id": "alarm.query",
                "input_bindings": {"device_id": {"source": "task", "key": "device_id"}},
            },
            {
                "state_id": "verify_after",
                "kind": "agent",
                "agent_id": "noc-diagnosis-agent",
                "capability_id": "alarm.verify",
                "input_bindings": {
                    "device_id": {"source": "task", "key": "device_id"},
                    "alarms": {"source": "state", "state_id": "query_after", "key": "alarms"},
                },
            },
            {"state_id": "complete", "kind": "terminal"},
            {"state_id": "manual_review", "kind": "terminal"},
        ],
        "transitions": [
            {"from_state": "query_before", "condition": "success", "to_state": "diagnose_before"},
            {"from_state": "query_before", "condition": "failure", "to_state": "manual_review"},
            {"from_state": "query_before", "condition": "timeout", "to_state": "manual_review"},
            {"from_state": "diagnose_before", "condition": "resolved", "to_state": "complete"},
            {"from_state": "diagnose_before", "condition": "needs_remediation", "to_state": "approve_restart"},
            {"from_state": "diagnose_before", "condition": "manual_review", "to_state": "manual_review"},
            {"from_state": "diagnose_before", "condition": "failure", "to_state": "manual_review"},
            {"from_state": "diagnose_before", "condition": "timeout", "to_state": "manual_review"},
            {"from_state": "approve_restart", "condition": "approved", "to_state": "restart_interface"},
            {"from_state": "approve_restart", "condition": "rejected", "to_state": "manual_review"},
            {"from_state": "approve_restart", "condition": "expired", "to_state": "manual_review"},
            {"from_state": "restart_interface", "condition": "success", "to_state": "query_after"},
            {"from_state": "restart_interface", "condition": "failure", "to_state": "manual_review"},
            {"from_state": "restart_interface", "condition": "timeout", "to_state": "manual_review"},
            {"from_state": "query_after", "condition": "success", "to_state": "verify_after"},
            {"from_state": "query_after", "condition": "failure", "to_state": "manual_review"},
            {"from_state": "query_after", "condition": "timeout", "to_state": "manual_review"},
            {"from_state": "verify_after", "condition": "resolved", "to_state": "complete"},
            {"from_state": "verify_after", "condition": "unresolved", "to_state": "manual_review"},
            {"from_state": "verify_after", "condition": "manual_review", "to_state": "manual_review"},
            {"from_state": "verify_after", "condition": "failure", "to_state": "manual_review"},
            {"from_state": "verify_after", "condition": "timeout", "to_state": "manual_review"},
        ],
        "terminal_states": ["complete", "manual_review"],
    }


def test_valid_fsm_is_verified_and_hashed():
    result = validate_fsm(_valid_fsm(), default_agent_cards())
    assert result.valid is True
    assert result.fsm is not None
    assert len(result.plan_hash or "") == 64
    assert result.fsm.plan_hash == result.plan_hash
    assert result.fsm.agent_registry_hash == result.agent_registry_hash


def test_unknown_agent_capability_and_missing_input_fail_closed():
    candidate = _valid_fsm()
    candidate["states"][0]["agent_id"] = "unregistered-agent"
    candidate["states"][1]["capability_id"] = "alarm.unknown"
    candidate["task"]["inputs"].pop("device_id")
    result = validate_fsm(candidate, default_agent_cards())
    codes = {issue.code for issue in result.issues}
    assert result.valid is False
    assert {"unknown_agent", "unknown_capability", "missing_task_input"} <= codes


def test_cycle_and_missing_failure_timeout_path_are_rejected():
    candidate = _valid_fsm()
    candidate["transitions"].append(
        {"from_state": "manual_review", "condition": "failure", "to_state": "query_before"}
    )
    candidate["transitions"] = [
        transition
        for transition in candidate["transitions"]
        if not (
            transition["from_state"] == "query_before"
            and transition["condition"] in {"failure", "timeout"}
        )
    ]
    result = validate_fsm(candidate, default_agent_cards())
    codes = {issue.code for issue in result.issues}
    assert result.valid is False
    assert "cycle" in codes
    assert "missing_failure_timeout_path" in codes


def test_high_risk_state_cannot_bypass_approval():
    candidate = _valid_fsm()
    candidate["transitions"].append(
        {"from_state": "diagnose_before", "condition": "needs_remediation", "to_state": "restart_interface"}
    )
    result = validate_fsm(candidate, default_agent_cards())
    assert result.valid is False
    assert any(issue.code == "approval_bypass" for issue in result.issues)


def test_model_view_and_registry_hash_are_stable_and_redacted():
    cards = [
        {
            **default_agent_cards()[0].to_dict(),
            "endpoint": "https://internal.invalid",
            "api_key": "do-not-expose",
            "system_prompt": "private",
        }
    ]
    view = agent_cards_model_view(cards)
    assert "endpoint" not in view[0]
    assert "api_key" not in view[0]
    assert "system_prompt" not in view[0]
    assert agent_registry_hash(cards) == agent_registry_hash(list(reversed(cards)))


def test_plan_and_history_snapshot_include_only_redacted_registry():
    registry = agent_cards_model_view(default_agent_cards())
    plan = SimpleNamespace(
        plan_id="plan_test",
        title="NOC plan",
        description="",
        task_input="query",
        status="draft",
        total_steps=0,
        completed_steps=0,
        result_summary=None,
        steps=[],
        created_at=None,
        updated_at=None,
        extra_data={"fsm": _valid_fsm(), "agent_registry": registry},
    )

    serialized = PlanService.plan_to_dict(plan)
    snapshot = PlanService.build_execution_snapshot(
        plan,
        completed_steps=0,
        total_steps=0,
        result_text="done",
    )
    assert serialized["agent_registry"] == registry
    assert snapshot["agent_registry"] == registry
    payload = json.dumps({"serialized": serialized, "snapshot": snapshot})
    assert "api_key" not in payload
    assert "endpoint" not in payload
    assert "system_prompt" not in payload


def test_capability_proposal_is_expanded_into_verified_fsm():
    proposal = {
        "task": {
            "goal": "repair device",
            "inputs": {
                "device_id": "NE-1",
                "interface": "ge-0/0/1",
                "reason": "test",
                "title": "repair",
                "severity": "major",
            },
        },
        "selected_capabilities": [
            {"agent_id": "noc-diagnosis-agent", "capability_id": "alarm.query"},
            {"agent_id": "noc-diagnosis-agent", "capability_id": "alarm.diagnose"},
            {"agent_id": "config-change-agent", "capability_id": "interface.restart"},
            {"agent_id": "noc-diagnosis-agent", "capability_id": "alarm.verify"},
        ],
    }

    built = build_fsm_from_proposal(proposal, default_agent_cards())
    assert built.valid and built.fsm is not None
    state_capabilities = [
        state.get("capability_id")
        for state in built.fsm["states"]
        if state["kind"] == "agent"
    ]
    assert state_capabilities == [
        "alarm.query",
        "alarm.diagnose",
        "incident.create",
        "interface.restart",
        "alarm.verify",
    ]
    assert any(state["kind"] == "approval" for state in built.fsm["states"])
    verified = validate_fsm(built.fsm, default_agent_cards())
    assert verified.valid


def test_capability_proposal_reports_missing_transitive_input():
    proposal = {
        "task": {"goal": "restart", "inputs": {"device_id": "NE-1", "interface": "ge-0/0/1"}},
        "selected_capabilities": [
            {"agent_id": "config-change-agent", "capability_id": "interface.restart"}
        ],
    }

    built = build_fsm_from_proposal(proposal, default_agent_cards())
    assert not built.valid
    assert {issue.code for issue in built.issues} == {"missing_task_input"}


def _ticket_fsm(severity_value):
    return {
        "schema_version": "fsm.v1",
        "task": {
            "goal": "create ticket",
            "inputs": {
                "device_id": "NE-1",
                "title": "t",
                "severity": severity_value,
                "summary": "s",
            },
        },
        "initial_state": "create_ticket",
        "states": [
            {
                "state_id": "create_ticket",
                "kind": "agent",
                "agent_id": "config-change-agent",
                "capability_id": "incident.create",
                "input_bindings": {
                    "device_id": {"source": "task", "key": "device_id"},
                    "title": {"source": "task", "key": "title"},
                    "severity": {"source": "task", "key": "severity"},
                    "summary": {"source": "task", "key": "summary"},
                },
            },
            {"state_id": "complete", "kind": "terminal"},
            {"state_id": "manual_review", "kind": "terminal"},
        ],
        "transitions": [
            {"from_state": "create_ticket", "condition": "success", "to_state": "complete"},
            {"from_state": "create_ticket", "condition": "failure", "to_state": "manual_review"},
            {"from_state": "create_ticket", "condition": "timeout", "to_state": "manual_review"},
        ],
        "terminal_states": ["complete", "manual_review"],
    }


def test_task_input_enum_violation_rejected():
    result = validate_fsm(_ticket_fsm("high"), default_agent_cards())
    assert not result.valid
    assert any(issue.code == "invalid_task_input_value" for issue in result.issues)


def test_task_input_enum_conforming_accepted():
    result = validate_fsm(_ticket_fsm("major"), default_agent_cards())
    assert result.valid and result.fsm is not None


def test_task_input_type_violation_rejected():
    fsm = _valid_fsm()
    fsm["task"]["inputs"]["ticket_id"] = 12345
    result = validate_fsm(fsm, default_agent_cards())
    assert not result.valid
    assert any(issue.code == "invalid_task_input_value" for issue in result.issues)
