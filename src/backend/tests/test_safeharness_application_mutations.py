"""Security checks for SafeHarness-controlled application writes."""

import asyncio
import ast
import hashlib
import hmac
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from core.infra.exceptions import AccessDeniedError
from core.safeharness_capabilities import reject_uncontrolled_mutation
from core.safeharness_mutations import MutationPrepareRequest, PendingMutation
from fastapi import HTTPException
from pydantic import ValidationError

_ROUTES = Path(__file__).resolve().parents[1] / "api" / "routes" / "v1"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _ROUTES / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


catalog = _load("application_catalog_contract", "catalog.py")
sys.modules["api.routes.v1.catalog"] = catalog
safeharness = _load("application_safeharness_contract", "safeharness.py")


def _pending(action, target):
    return PendingMutation(
        approval_id="a" * 32,
        decision_id="decision-1",
        operator_id="user-1",
        tenant_id="tenant-a",
        action=action,
        target_values=target,
        reason="Controlled application change",
        idempotency_key="mutation-key-0001",
        request_digest="digest",
        expires_at=999999,
    )


def _authorization():
    return {
        "authorization_id": "b" * 64,
        "audit_reference": "application-authorization:" + "b" * 64,
        "envelope_digest": "c" * 64,
    }


def test_legacy_application_write_routes_fail_closed():
    with pytest.raises(AccessDeniedError):
        reject_uncontrolled_mutation()
    for filename in ("catalog.py", "plugins.py", "automations.py"):
        source = (_ROUTES / filename).read_text(encoding="utf-8")
        assert "reject_uncontrolled_mutation()" in source


def test_unsupported_security_writes_are_guarded_at_entry():
    guarded = {
        "plugins.py": {"init_feishu_app", "reset_feishu_app", "install_plugin", "import_plugin", "uninstall_plugin", "set_enabled"},
        "automations.py": {"create_automation", "update_automation", "delete_automation", "pause_automation", "resume_automation", "trigger_automation"},
        "me_capabilities.py": {"create_my_mcp_server", "delete_my_mcp_server", "upload_my_skill", "create_my_skill", "save_my_skill_file", "delete_my_skill_file", "upload_my_skill_file", "delete_my_skill", "set_my_skill_icon"},
        "marketplace.py": {"install_skill", "submit_skill", "withdraw_submission"},
        "mcp_marketplace.py": {"start_oauth", "oauth_callback", "install_item", "submit_item", "withdraw_submission"},
        "agent_marketplace.py": {"install_agent", "submit_agent", "withdraw_submission"},
    }
    for filename, names in guarded.items():
        tree = ast.parse((_ROUTES / filename).read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in names:
            body = functions[name].body
            first = body[1] if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str) else body[0]
            assert isinstance(first, ast.Expr)
            assert isinstance(first.value, ast.Call)
            assert isinstance(first.value.func, ast.Name)
            assert first.value.func.id == "reject_uncontrolled_mutation"


def test_application_mutation_target_schema_is_action_specific():
    body = {
        "schema_version": "1.0",
        "action": "capability.enable",
        "target": {"tenant_id": "tenant-a", "kind": "skill", "item_id": "skill-1"},
        "reason": "Controlled application change",
        "idempotency_key": "mutation-key-0001",
        "reauth": {"method": "password", "credential": "password"},
    }
    assert MutationPrepareRequest.model_validate(body).target.item_id == "skill-1"
    with pytest.raises(ValidationError):
        MutationPrepareRequest.model_validate(
            {**body, "target": {**body["target"], "install_id": "plugin-1"}}
        )


def test_controlled_capability_mutation_uses_authoritative_user(monkeypatch):
    called = {}

    async def update(**kwargs):
        called.update(kwargs)
        return {"data": {"kind": kwargs["kind"], "id": kwargs["id"], "enabled": True}}

    monkeypatch.setattr(catalog, "_update_catalog_item_for_user", update)
    async def authorize(_item):
        return _authorization()
    monkeypatch.setattr(safeharness, "_authorize_application_mutation", authorize)
    result = asyncio.run(
        safeharness._execute_sumc_mutation(
            _pending("capability.enable", {"kind": "skill", "item_id": "skill-1"}),
            object(),
        )
    )

    assert called["user_id"] == "user-1"
    assert called["kind"] == "skill"
    assert called["id"] == "skill-1"
    assert result["status"] == "completed"
    assert result["receipt"]["source"] == "safeharness_application_authorization"
    assert result["receipt"]["authorization_id"] == "b" * 64
    assert result["receipt"]["gateway_verified"] is False


def test_application_authorization_binds_full_envelope(monkeypatch):
    item = _pending("plugin.disable", {"install_id": "plugin-1"})
    envelope = safeharness._application_authorization_envelope(item)
    encoded = json.dumps(
        envelope, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    key = "m" * 32
    signature = hmac.new(
        key.encode(), b"safeharness-application-authorization-v1\0" + encoded, hashlib.sha256
    ).hexdigest()

    async def authorize(method, route, **kwargs):
        assert (method, route, kwargs["payload"]) == ("POST", "application_authorize", envelope)
        return {
            "schema_version": "1.0",
            "operation": "authorize_application_mutation",
            "action": item.action,
            "authorization_id": signature,
            "signature": signature,
            "envelope_digest": hashlib.sha256(encoded).hexdigest(),
            "audit_reference": "application-authorization:" + signature,
            "integrity": "hmac_envelope_signature",
            "persistent": False,
            "gateway_verified": False,
            "production_ready": False,
        }

    monkeypatch.setattr(safeharness, "_upstream_request", authorize)
    monkeypatch.setattr(
        safeharness, "settings", SimpleNamespace(safeharness=SimpleNamespace(mutation_key=key))
    )
    result = asyncio.run(safeharness._authorize_application_mutation(item))
    assert result["signature"] == signature


def test_invalid_application_authorization_fails_before_local_write(monkeypatch):
    async def invalid(*_args, **_kwargs):
        return {
            "schema_version": "1.0",
            "operation": "authorize_application_mutation",
            "action": "capability.enable",
            "authorization_id": "0" * 64,
            "signature": "0" * 64,
            "envelope_digest": "0" * 64,
            "audit_reference": "application-authorization:" + "0" * 64,
            "integrity": "hmac_envelope_signature",
            "persistent": False,
            "gateway_verified": False,
            "production_ready": False,
        }

    monkeypatch.setattr(safeharness, "_upstream_request", invalid)
    monkeypatch.setattr(
        safeharness,
        "settings",
        SimpleNamespace(safeharness=SimpleNamespace(mutation_key="m" * 32)),
    )
    with pytest.raises(safeharness.SafeHarnessUpstreamError):
        asyncio.run(
            safeharness._authorize_application_mutation(
                _pending("capability.enable", {"kind": "skill", "item_id": "skill-1"})
            )
        )


def test_control_plane_failure_prevents_local_application_write(monkeypatch):
    async def unavailable(_item):
        raise safeharness.SafeHarnessUpstreamError("unavailable")

    async def should_not_write(**_kwargs):
        raise AssertionError("local write must not run")

    monkeypatch.setattr(safeharness, "_authorize_application_mutation", unavailable)
    monkeypatch.setattr(catalog, "_update_catalog_item_for_user", should_not_write)
    with pytest.raises(safeharness.SafeHarnessUpstreamError):
        asyncio.run(
            safeharness._execute_sumc_mutation(
                _pending("capability.enable", {"kind": "skill", "item_id": "skill-1"}),
                object(),
            )
        )


def test_controlled_automation_fails_closed_without_scheduler(monkeypatch):
    class Service:
        def __init__(self, _db):
            pass

        def get_task(self, task_id, user_id):
            assert (task_id, user_id) == ("task-1", "user-1")
            return SimpleNamespace(task_id=task_id, user_id=user_id, status="active")

    automation_service = ModuleType("core.services.automation_service")
    automation_service.AutomationService = Service
    scheduler_module = ModuleType("orchestration.schedulers.automation_scheduler")
    scheduler_module.get_scheduler = lambda: None
    monkeypatch.setitem(sys.modules, automation_service.__name__, automation_service)
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    async def authorize(_item):
        return _authorization()
    monkeypatch.setattr(safeharness, "_authorize_application_mutation", authorize)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            safeharness._execute_sumc_mutation(
                _pending("automation.run", {"task_id": "task-1"}), object()
            )
        )

    assert exc.value.status_code == 503
