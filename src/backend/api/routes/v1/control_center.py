"""Safe, user-scoped test runs and AgentCard configuration workflow."""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any, Literal

from core.auth.backend import UserContext, get_current_user
from core.db.engine import get_db
from core.db.models import ConfigCenterDraft, ConfigCenterVersion, SafeHarnessTestRun, UserAgent
from core.infra.responses import error_response, sse_response, success_response
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

router = APIRouter(prefix="/v1/config-center", tags=["Config Center"])
test_router = APIRouter(prefix="/v1/safeharness", tags=["SafeHarness Test Center"])

SCENARIOS = (
    ("input-injection", "输入注入阻断", "input_guard"),
    ("nested-tool-result", "嵌套工具结果阻断", "post_tool_guard"),
    ("restored-tool-call", "恢复消息工具调用清除", "restored_message_guard"),
    ("restored-history", "恢复历史重新检查", "restored_history_guard"),
)
FORBIDDEN_KEYS = {
    "api_key", "authorization", "credential", "endpoint", "handler", "password",
    "secret", "system_prompt", "token", "mutation_key", "admin_key", "signing_key",
}


def _tenant(user: UserContext, db: Session) -> str:
    from core.db.models import UserShadow

    shadow = db.get(UserShadow, str(user.user_id))
    metadata = shadow.extra_data if shadow and isinstance(shadow.extra_data, dict) else {}
    return str(metadata.get("tenant_id") or user.user_center_id or user.user_id)


def _walk_forbidden(value: Any, path: str = "config") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                return f"{path}.{key}"
            found = _walk_forbidden(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _walk_forbidden(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _next_version(current: str) -> str:
    if current.startswith("V"):
        major = current[1:].split(".", 1)[0]
        if major.isdigit():
            return f"V{int(major) + 1}.0"
    return f"V{uuid.uuid4().hex[:8]}"


def _diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            next_path = f"{path}.{key}" if path else str(key)
            if key not in before:
                changes.append({"path": next_path, "kind": "added", "after": after[key]})
            elif key not in after:
                changes.append({"path": next_path, "kind": "removed", "before": before[key]})
            else:
                changes.extend(_diff(before[key], after[key], next_path))
        return changes
    if before != after:
        return [{"path": path or "$", "kind": "changed", "before": before, "after": after}]
    return []


def _agent_card(agent: UserAgent) -> dict[str, Any]:
    extra = agent.extra_config if isinstance(agent.extra_config, dict) else {}
    card = extra.get("agent_card")
    return copy.deepcopy(card) if isinstance(card, dict) else {}


def _planning(agent: UserAgent) -> dict[str, Any]:
    extra = agent.extra_config if isinstance(agent.extra_config, dict) else {}
    planning = extra.get("planning")
    return copy.deepcopy(planning) if isinstance(planning, dict) else {}


def _current_payload(agent: UserAgent) -> dict[str, Any]:
    return {"agent_card": _agent_card(agent), "planning": _planning(agent)}


def _validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if set(payload) - {"agent_card", "planning"}:
        return None, {"valid": False, "reason_code": "config_unknown_field"}
    forbidden = _walk_forbidden(payload)
    if forbidden:
        return None, {"valid": False, "reason_code": "config_sensitive_field", "field": forbidden}
    normalized = {"agent_card": None, "planning": {}}
    if payload.get("agent_card") is not None:
        from orchestration.subagents.fsm_contracts import normalize_agent_card

        try:
            normalized["agent_card"] = normalize_agent_card(payload["agent_card"])
        except (TypeError, ValueError) as exc:
            return None, {"valid": False, "reason_code": "agent_card_invalid", "message": str(exc)[:300]}
    if payload.get("planning") is not None:
        if not isinstance(payload["planning"], dict):
            return None, {"valid": False, "reason_code": "planning_invalid"}
        normalized["planning"] = copy.deepcopy(payload["planning"])
        allowed = {"enabled", "mode", "max_retries", "approval_required", "fsm"}
        unknown = set(normalized["planning"]) - allowed
        if unknown:
            return None, {"valid": False, "reason_code": "planning_unknown_field", "fields": sorted(unknown)}
        if "fsm" in normalized["planning"]:
            from orchestration.subagents.fsm_contracts import normalize_fsm

            try:
                normalized["planning"]["fsm"] = normalize_fsm(normalized["planning"]["fsm"])
            except (TypeError, ValueError) as exc:
                return None, {"valid": False, "reason_code": "fsm_invalid", "message": str(exc)[:300]}
    return normalized, {"valid": True, "reason_code": "config_valid"}


def _draft_view(item: ConfigCenterDraft) -> dict[str, Any]:
    return {
        "draft_id": item.draft_id,
        "config_key": item.config_key,
        "agent_id": item.agent_id,
        "base_version": item.base_version,
        "payload": item.payload,
        "validation": item.validation,
        "status": item.status,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


class TestRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: Literal["input-injection", "nested-tool-result", "restored-tool-call", "restored-history", "all"]
    parameters: dict[str, Any] = Field(default_factory=dict)


class ConfigValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any]


class DraftCreateRequest(ConfigValidateRequest):
    base_version: str = Field(min_length=1, max_length=64)


class DraftActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"] = "approve"


@test_router.get("/test-scenarios")
def list_test_scenarios(_: UserContext = Depends(get_current_user)):
    return success_response(data={
        "schema_version": "1.0",
        "production_ready": False,
        "scenarios": [
            {"scenario_id": scenario_id, "name": name, "blocked_stage": stage,
             "input_schema": {"type": "object", "additionalProperties": False},
             "expected": {"passed": True, "model_calls": 0 if scenario_id != "nested-tool-result" else 2},
             "capability": "deterministic_safeharness_demo"}
            for scenario_id, name, stage in SCENARIOS
        ],
    })


async def _run_scenarios(scenario_id: str) -> list[dict[str, Any]]:
    from examples.langgraph_noc.security_demo import run_security_scenarios

    results = await run_security_scenarios(scenario_id)
    return [
        {
            "name": item.get("name"),
            "blocked_stage": item.get("blocked_stage"),
            "passed": item.get("passed") is True,
            "evidence": item.get("evidence") if isinstance(item.get("evidence"), dict) else {},
        }
        for item in results
    ]


@test_router.post("/test-runs")
async def create_test_run(
    body: TestRunRequest,
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        results = await _run_scenarios(body.scenario_id)
    except Exception as exc:  # deterministic runner failure is an explicit test failure
        results = [{"name": body.scenario_id, "blocked_stage": "runner", "passed": False,
                    "evidence": {"reason_code": "test_runner_failed"}}]
    run_id = f"test_{uuid.uuid4().hex[:20]}"
    payload = {"run_id": run_id, "scenario_id": body.scenario_id, "status": "pass" if all(item["passed"] for item in results) else "fail",
               "expected": {"production_ready": False}, "observed": results,
               "reason_code": "deterministic_pass" if all(item["passed"] for item in results) else "deterministic_failed",
               "production_ready": False}
    db.add(SafeHarnessTestRun(run_id=run_id, user_id=str(user.user_id), tenant_id=_tenant(user, db), scenario_id=body.scenario_id, status=payload["status"], result=payload))
    db.commit()
    return success_response(data=payload)


@test_router.get("/test-runs/{run_id}")
def get_test_run(run_id: str, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    item = db.get(SafeHarnessTestRun, run_id)
    if not item or item.user_id != str(user.user_id):
        return error_response(code=404, message="Test run not found", status_code=404)
    return success_response(data=item.result)


@test_router.get("/test-runs/{run_id}/events")
async def get_test_run_events(run_id: str, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    item = db.get(SafeHarnessTestRun, run_id)
    if not item or item.user_id != str(user.user_id):
        raise HTTPException(status_code=404, detail="Test run not found")

    async def events():
        yield f"event: test_started\ndata: {json.dumps({'run_id': run_id}, ensure_ascii=False)}\n\n"
        yield f"event: test_finished\ndata: {json.dumps(item.result, ensure_ascii=False)}\n\n"

    return sse_response(events())


@router.get("/catalog")
def config_catalog(_: UserContext = Depends(get_current_user)):
    return success_response(data={"schema_version": "1.0", "groups": [{
        "key": "agent", "label": "Agent 配置", "scope": "user", "editable": True,
        "fields": {"agent_card": {"type": "json", "sensitive": False}, "planning": {"type": "object", "sensitive": False}},
    }]})


@router.get("/state")
def config_state(user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    agents = db.query(UserAgent).filter(UserAgent.user_id == str(user.user_id), UserAgent.owner_type == "user").all()
    return success_response(data={"schema_version": "1.0", "versions": [{
        "config_key": "agent", "agent_id": agent.agent_id,
        "version": str((agent.extra_config or {}).get("version") or "V1.0"),
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
    } for agent in agents]})


@router.post("/validate")
def validate_config(body: ConfigValidateRequest, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    agent = db.get(UserAgent, body.agent_id)
    if not agent or agent.user_id != str(user.user_id):
        return error_response(code=404, message="Agent not found", status_code=404)
    normalized, validation = _validate_payload(body.payload)
    return success_response(data={"valid": validation["valid"], "validation": validation, "normalized": normalized})


@router.post("/drafts")
def create_draft(body: DraftCreateRequest, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    agent = db.get(UserAgent, body.agent_id)
    if not agent or agent.user_id != str(user.user_id):
        return error_response(code=404, message="Agent not found", status_code=404)
    current = _current_payload(agent)
    current_version = str((agent.extra_config or {}).get("version") or "V1.0")
    if body.base_version != current_version:
        return error_response(code=409, message="Configuration base version is stale", data={"code": "config_base_version_conflict"}, status_code=409)
    normalized, validation = _validate_payload(body.payload)
    if not validation["valid"] or normalized is None:
        return error_response(code=422, message="Configuration validation failed", data={"validation": validation}, status_code=422)
    item = ConfigCenterDraft(draft_id=f"draft_{uuid.uuid4().hex[:20]}", user_id=str(user.user_id), tenant_id=_tenant(user, db), config_key="agent", agent_id=body.agent_id, base_version=body.base_version, payload=normalized, validation=validation, status="draft")
    db.add(item)
    db.commit()
    return success_response(data={**_draft_view(item), "diff": _diff(current, normalized)})


def _owned_draft(draft_id: str, user: UserContext, db: Session) -> ConfigCenterDraft:
    item = db.get(ConfigCenterDraft, draft_id)
    from core.db.models import UserShadow

    shadow = db.get(UserShadow, str(user.user_id))
    metadata = shadow.extra_data if shadow and isinstance(shadow.extra_data, dict) else {}
    is_admin = metadata.get("role") == "super_admin" and str(metadata.get("tenant_id") or "") == item.tenant_id if item else False
    if not item or (item.user_id != str(user.user_id) and not is_admin):
        raise HTTPException(status_code=404, detail="Draft not found")
    return item


@router.get("/drafts/{draft_id}")
def get_draft(draft_id: str, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    item = _owned_draft(draft_id, user, db)
    agent = db.get(UserAgent, item.agent_id) if item.agent_id else None
    return success_response(data={**_draft_view(item), "diff": _diff(_current_payload(agent), item.payload) if agent else []})


@router.post("/drafts/{draft_id}/submit")
def submit_draft(draft_id: str, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    item = _owned_draft(draft_id, user, db)
    if item.status != "draft":
        return error_response(code=409, message="Draft is not submit-ready", status_code=409)
    item.status = "pending_approval"
    db.commit()
    return success_response(data=_draft_view(item))


@router.post("/drafts/{draft_id}/approve")
def approve_draft(draft_id: str, body: DraftActionRequest, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    item = _owned_draft(draft_id, user, db)
    shadow_tenant = _tenant(user, db)
    from core.db.models import UserShadow
    shadow = db.get(UserShadow, str(user.user_id))
    role = (shadow.extra_data or {}).get("role") if shadow else None
    if role != "super_admin":
        return error_response(code=403, message="Configuration approval requires super_admin", status_code=403)
    if item.status != "pending_approval":
        return error_response(code=409, message="Draft is not pending approval", status_code=409)
    item.status = "approved" if body.decision == "approve" else "rejected"
    db.commit()
    return success_response(data=_draft_view(item))


@router.post("/drafts/{draft_id}/publish")
def publish_draft(draft_id: str, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    item = _owned_draft(draft_id, user, db)
    if item.status != "approved":
        return error_response(code=409, message="Draft must be approved before publish", status_code=409)
    agent = db.get(UserAgent, item.agent_id)
    if not agent:
        return error_response(code=404, message="Agent not found", status_code=404)
    current_version = str((agent.extra_config or {}).get("version") or "V1.0")
    if current_version != item.base_version:
        return error_response(code=409, message="Configuration changed after draft creation", status_code=409)
    next_version = _next_version(current_version)
    extra = dict(agent.extra_config or {})
    extra.update(item.payload)
    extra["version"] = next_version
    agent.extra_config = extra
    item.status = "published"
    audit_reference = f"config_publish_{uuid.uuid4().hex[:16]}"
    version = ConfigCenterVersion(version_id=f"ver_{uuid.uuid4().hex[:20]}", user_id=item.user_id, tenant_id=item.tenant_id, config_key=item.config_key, agent_id=item.agent_id, version=next_version, payload=item.payload, source="publish", audit_reference=audit_reference)
    db.add(version)
    db.commit()
    return success_response(data={"draft": _draft_view(item), "version": next_version, "audit_reference": audit_reference})


@router.get("/history")
def config_history(user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    items = db.query(ConfigCenterVersion).filter(ConfigCenterVersion.user_id == str(user.user_id)).order_by(ConfigCenterVersion.created_at.desc()).limit(100).all()
    return success_response(data={"versions": [{"version_id": item.version_id, "agent_id": item.agent_id, "version": item.version, "source": item.source, "audit_reference": item.audit_reference, "created_at": item.created_at.isoformat() if item.created_at else None} for item in items]})


@router.post("/rollback")
def rollback_config(version_id: str, user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    source = db.get(ConfigCenterVersion, version_id)
    if not source or source.user_id != str(user.user_id):
        return error_response(code=404, message="Configuration version not found", status_code=404)
    agent = db.get(UserAgent, source.agent_id)
    if not agent:
        return error_response(code=404, message="Agent not found", status_code=404)
    current = str((agent.extra_config or {}).get("version") or "V1.0")
    next_version = _next_version(current)
    extra = dict(agent.extra_config or {})
    extra.update(source.payload)
    extra["version"] = next_version
    agent.extra_config = extra
    audit_reference = f"config_rollback_{uuid.uuid4().hex[:16]}"
    db.add(ConfigCenterVersion(version_id=f"ver_{uuid.uuid4().hex[:20]}", user_id=str(user.user_id), tenant_id=source.tenant_id, config_key=source.config_key, agent_id=source.agent_id, version=next_version, payload=source.payload, source="rollback", audit_reference=audit_reference))
    db.commit()
    return success_response(data={"version": next_version, "audit_reference": audit_reference, "source_version": source.version})


__all__ = ["router", "test_router"]
