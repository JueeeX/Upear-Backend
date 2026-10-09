"""Contract checks for fail-closed SafeHarness discovery and BFF routes."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

_API_ROOT = Path(__file__).resolve().parents[1] / "api"
_SAFEHARNESS_PATH = _API_ROOT / "routes" / "v1" / "safeharness.py"
_SPEC = importlib.util.spec_from_file_location("safeharness_routes_contract", _SAFEHARNESS_PATH)
assert _SPEC and _SPEC.loader
safeharness_routes = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = safeharness_routes
_SPEC.loader.exec_module(safeharness_routes)

from core.auth.backend import UserContext
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from orchestration.subagents.fsm_contracts import ContractError, normalize_agent_card


def _user() -> UserContext:
    return UserContext(user_id="user-1", user_center_id="center-1", username="operator")


class _DB:
    def __init__(self, metadata=None):
        self.shadow = SimpleNamespace(extra_data=metadata) if metadata is not None else None

    def get(self, _model, _key):
        return self.shadow


def _principal(*, role="super_admin", tenant_id="tenant-a", can_view=True):
    return safeharness_routes.SafeHarnessPrincipal(
        user_id="user-1",
        tenant_id=tenant_id,
        role=role,
        can_view=can_view,
        can_manage=role == "super_admin",
    )


def _agent_card(**overrides):
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


def test_agent_card_normalization_rejects_sensitive_fields():
    with pytest.raises(ContractError, match="api_key"):
        normalize_agent_card(_agent_card(api_key="secret"))


def test_agent_card_normalization_is_deterministic():
    card = _agent_card(capabilities=[{**_agent_card()["capabilities"][0], "outcomes": ["z", "a"]}])
    normalized = normalize_agent_card(card)
    assert normalized["schema_version"] == "agent-card.v1"
    assert normalized["capabilities"][0]["outcomes"] == ["a", "z"]


def _mutation_body(**overrides):
    value = {
        "schema_version": "1.0",
        "action": "agent.quarantine",
        "target": {"tenant_id": "tenant-a", "agent_id": "agent-1"},
        "reason": "Incident containment",
        "idempotency_key": "mutation-key-0001",
        "reauth": {"method": "password", "credential": "operator-password"},
    }
    value.update(overrides)
    return value


def _request(query=b"", body=b"", headers=None) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers or [],
            "query_string": query,
        },
        receive,
    )


def _test_client(principal=None, *, status_code=None) -> TestClient:
    app = FastAPI()
    app.include_router(safeharness_routes.router)

    def identity():
        if status_code is not None:
            raise HTTPException(status_code=status_code, detail="blocked")
        return principal or _principal()

    app.dependency_overrides[safeharness_routes._principal_dependency] = identity
    app.dependency_overrides[safeharness_routes.get_db] = lambda: _DB()
    return TestClient(app)


@pytest.fixture
def configured_bff(monkeypatch):
    config = SimpleNamespace(
        admin_ui_url="http://127.0.0.1:38308",
        ui_access_token="u" * 32,
        mutation_key="m" * 32,
        timeout_seconds=5.0,
        max_response_bytes=1024,
    )
    monkeypatch.setattr(safeharness_routes, "settings", SimpleNamespace(safeharness=config))
    return config


@pytest.fixture(autouse=True)
def reset_protected_chat_handler():
    safeharness_routes.unregister_protected_chat_stream()
    safeharness_routes.mutation_coordinator.reset_for_tests()
    yield
    safeharness_routes.unregister_protected_chat_stream()
    safeharness_routes.mutation_coordinator.reset_for_tests()


def _chat_event(context, event_type, sequence, data):
    return {
        "schema_version": "1.0",
        "event_id": f"event-{sequence}",
        "type": event_type,
        "sequence": sequence,
        "tenant_id": context["tenant_id"],
        "user_id": context["user_id"],
        "agent_id": context["agent_id"],
        "conversation_id": context["conversation_id"],
        "session_id": context["session_id"],
        "run_id": context["run_id"],
        "trace_id": context["trace_id"],
        "timestamp": "2026-08-10T00:00:00+00:00",
        "data": data,
    }


def _sse_payloads(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_safeharness_capability_route_is_registered():
    source = (_API_ROOT / "routes" / "v1" / "meta.py").read_text(encoding="utf-8")
    assert '@router.get("/safeharness"' in source
    assert "conversation_id=conversation_id" in source
    assert "agent_id=agent_id" in source
    assert "project_id=project_id" in source

    registry = (_API_ROOT / "routes" / "v1" / "__init__.py").read_text(encoding="utf-8")
    assert '("safeharness", "router")' in registry


def test_safeharness_capability_contract_fails_closed_without_bff():
    data = asyncio.run(safeharness_routes.capability_discovery_payload(None, None))

    assert data["available"] is False
    assert data["capabilities"] == {}
    assert data["context"]["user_id"] is None
    assert "admin_key" not in str(data).lower()


def test_only_named_safeharness_routes_are_registered():
    assert {route.path for route in safeharness_routes.router.routes} == {
        "/v1/safeharness/overview",
        "/v1/safeharness/audit",
        "/v1/safeharness/chat/stream",
        "/v1/safeharness/security-report",
        "/v1/safeharness/policies",
        "/v1/safeharness/policies/simulate",
        "/v1/safeharness/mutations",
        "/v1/safeharness/mutations/{approval_id}/decision",
    }


def test_protected_chat_route_is_unavailable_without_registered_handler():
    with _test_client() as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            json={
                "conversation_id": "chat-1",
                "agent_id": "main_agent",
                "message": "Summarize the alarm.",
            },
        )

    assert response.status_code == 503
    assert response.json()["data"]["error"]["code"] == "safeharness_chat_unavailable"


def test_protected_chat_creates_first_session_and_persists_protection(monkeypatch):
    captured = {}

    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, _chat_id, _user_id):
            return captured.get("session")

        def ensure_session(self, **kwargs):
            captured["created"] = kwargs
            captured["session"] = SimpleNamespace(project_id=None)
            return captured["session"]

        def update_session(self, chat_id, user_id, fields):
            captured["updated"] = (chat_id, user_id, fields)

        def add_message(self, **_kwargs):
            return object()

    async def handler(context):
        yield _chat_event(context, "run_started", 1, {})
        yield _chat_event(context, "message", 2, {"content": "safe"})
        yield _chat_event(context, "run_finished", 3, {"status": "completed"})

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)
    safeharness_routes.register_protected_chat_stream(handler)
    with _test_client() as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            json={"conversation_id": "chat-new", "agent_id": "main_agent", "message": "hello"},
        )

    assert response.status_code == 200
    assert captured["created"]["extra_data"]["safeharness_protected"] is True
    assert captured["updated"] == (
        "chat-new",
        "user-1",
        {"extra_data": {"safeharness_protected": True}},
    )


def test_security_report_is_authenticated_and_redacted(monkeypatch, tmp_path):
    scenarios = []
    for name, (stage, description) in safeharness_routes._SECURITY_REPORT_PUBLIC_FIELDS.items():
        scenarios.append(
            {
                "name": name,
                "blocked_stage": stage,
                "description": description,
                "passed": True,
                "evidence": {"blocked": True},
            }
        )
    report = tmp_path / "security-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "overall_status": "pass",
                "production_ready": False,
                "scenarios": scenarios,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUMC_SECURITY_REPORT_PATH", str(report))

    with _test_client(status_code=401) as client:
        unauthorized = client.get("/v1/safeharness/security-report")
    with _test_client() as client:
        response = client.get("/v1/safeharness/security-report")

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["production_ready"] is False
    assert len(data["scenarios"]) == 4
    assert "api_key" not in response.text.lower()


def test_security_report_rejects_unknown_fields(monkeypatch, tmp_path):
    scenarios = []
    for name, (stage, description) in safeharness_routes._SECURITY_REPORT_PUBLIC_FIELDS.items():
        scenarios.append(
            {
                "name": name,
                "blocked_stage": stage,
                "description": description,
                "passed": True,
                "evidence": {"blocked": True},
                **({"raw_payload": "secret"} if name == "input-injection" else {}),
            }
        )
    report = tmp_path / "security-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "overall_status": "pass",
                "production_ready": False,
                "scenarios": scenarios,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUMC_SECURITY_REPORT_PATH", str(report))

    with _test_client() as client:
        response = client.get("/v1/safeharness/security-report")

    assert response.status_code == 503
    assert "secret" not in response.text


def test_protected_chat_stream_binds_authoritative_context(monkeypatch):
    captured = {}

    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, _chat_id, _user_id):
            return object()

        def get_session_with_access(self, chat_id, user_id):
            assert (chat_id, user_id) == ("chat-1", "user-1")
            return object()

        def add_message(self, **kwargs):
            captured.setdefault("messages", []).append(kwargs)
            return object()

    class UserAgentService:
        def __init__(self, _db):
            pass

        def get_by_id(self, agent_id, user_id=None):
            assert (agent_id, user_id) == ("agent-1", "user-1")
            return object()

    async def handler(context):
        captured.update(context)
        yield _chat_event(context, "run_started", 1, {})
        yield _chat_event(context, "message", 2, {"content": "safe response"})
        yield _chat_event(context, "run_finished", 3, {"status": "completed"})

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)
    monkeypatch.setattr(safeharness_routes, "UserAgentService", UserAgentService)
    safeharness_routes.register_protected_chat_stream(handler)
    with _test_client(_principal(role="member", can_view=True)) as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            headers={"X-Trace-ID": "trace-chat"},
            json={
                "conversation_id": "chat-1",
                "agent_id": "agent-1",
                "message": "Summarize the alarm.",
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Trace-ID"] == "trace-chat"
    payloads = _sse_payloads(response)
    assert [item["type"] for item in payloads] == [
        "run_started",
        "message",
        "run_finished",
    ]
    assert payloads[1]["data"] == {"content": "safe response"}
    assert captured["tenant_id"] == "tenant-a"
    assert captured["user_id"] == "user-1"
    assert captured["role"] == "member"
    assert captured["trace_id"] == "trace-chat"
    assert [item["role"] for item in captured["messages"]] == ["user", "assistant"]
    assert captured["messages"][1]["extra_data"]["protected"] is True


def test_protected_project_chat_binds_authoritative_project(monkeypatch):
    captured = {}

    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, chat_id, user_id):
            assert (chat_id, user_id) == ("chat-project", "user-1")
            return SimpleNamespace(project_id="project-1"), "admin"

        def add_message(self, **kwargs):
            captured.setdefault("messages", []).append(kwargs)
            return object()

    class ProjectService:
        def __init__(self, _db):
            pass

        def get_raw(self, project_id):
            assert project_id == "project-1"
            return SimpleNamespace(
                project_id=project_id,
                deleted_at=None,
                kind="personal",
                owner_user_id="user-1",
            )

    async def handler(context):
        captured.update(context)
        yield _chat_event(context, "run_started", 1, {})
        yield _chat_event(context, "message", 2, {"content": "safe project response"})
        yield _chat_event(context, "run_finished", 3, {"status": "completed"})

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)
    monkeypatch.setattr(safeharness_routes, "ProjectService", ProjectService)
    safeharness_routes.register_protected_chat_stream(handler)
    with _test_client(_principal(role="member", can_view=True)) as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            json={
                "conversation_id": "chat-project",
                "agent_id": "main_agent",
                "project_id": "project-1",
                "message": "Summarize this incident.",
            },
        )

    assert response.status_code == 200
    assert captured["project_id"] == "project-1"
    assert all(item["extra_data"]["project_id"] == "project-1" for item in captured["messages"])


def test_protected_project_chat_rejects_conversation_project_mismatch(monkeypatch):
    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, _chat_id, _user_id):
            return SimpleNamespace(project_id="project-1"), "admin"

    async def handler(_context):
        if False:
            yield {}

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)
    safeharness_routes.register_protected_chat_stream(handler)
    with _test_client(_principal(role="member", can_view=True)) as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            json={
                "conversation_id": "chat-project",
                "agent_id": "main_agent",
                "project_id": "project-2",
                "message": "Summarize this incident.",
            },
        )

    assert response.status_code == 404
    assert response.json()["data"]["error"]["code"] == "safeharness_chat_context_not_found"


def test_protected_chat_stream_accepts_redacted_security_timeline(monkeypatch):
    monkeypatch.setattr(safeharness_routes, "_validate_chat_scope", lambda *_args: None)
    captured = {}

    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, _chat_id, _user_id):
            return object()

        def add_message(self, **kwargs):
            captured.setdefault("messages", []).append(kwargs)
            return object()

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)

    def event(context, event_type, sequence, data):
        return _chat_event(context, event_type, sequence, data)

    async def handler(context):
        yield event(context, "run_started", 1, {})
        yield event(context, "policy_decision", 2, {
            "decision_id": "decision-1", "event_kind": "tool_call_proposed",
            "disposition": "require_approval", "policy_version": "v1", "reason_code": "review",
        })
        yield event(context, "tool_proposal", 3, {"decision_id": "decision-1", "tool_name": "restart_interface"})
        yield event(context, "approval", 4, {"approval_id": "approval-1", "status": "denied"})
        yield event(context, "execution", 5, {"decision_id": "decision-1", "tool_name": "restart_interface", "status": "failed"})
        yield event(context, "receipt", 6, {"receipt_id": "receipt-1", "decision_id": "decision-1", "enforced": True, "adapter": "langgraph"})
        yield event(context, "audit_reference", 7, {"record_type": "enforcement", "event_id": "event-1", "decision_id": "decision-1", "receipt_id": "receipt-1"})
        yield event(context, "blocked", 8, {"reason_code": "approval_denied", "message": "Request blocked."})
        yield event(context, "run_finished", 9, {"status": "blocked"})

    safeharness_routes.register_protected_chat_stream(handler)
    with _test_client() as client:
        response = client.post("/v1/safeharness/chat/stream", json={
            "conversation_id": "chat-1", "agent_id": "main_agent", "message": "Restart uplink."
        })

    assert response.status_code == 200
    assert [item["type"] for item in _sse_payloads(response)][1:8] == [
        "policy_decision", "tool_proposal", "approval", "execution", "receipt", "audit_reference", "blocked"
    ]
    assert "arguments" not in response.text
    assert "result" not in response.text
    timeline = captured["messages"][-1]["extra_data"]["security_timeline"]
    assert [item["type"] for item in timeline] == [
        "policy_decision", "tool_proposal", "approval", "execution",
        "receipt", "audit_reference", "blocked", "run_finished",
    ]
    assert timeline[-2]["data"]["reason_code"] == "approval_denied"


def test_protected_chat_route_runs_real_langgraph_bridge_with_fake_model(monkeypatch):
    root_src = Path(__file__).resolve().parents[4] / "src"
    if str(root_src) not in sys.path:
        sys.path.insert(0, str(root_src))
    from langchain_core.messages import AIMessage
    from safeharness.adapters.langgraph import LangGraphAgent
    from safeharness.chat_bridge import ProtectedChatRunner

    captured = []

    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, _chat_id, _user_id):
            return object()

        def add_message(self, **kwargs):
            captured.append(kwargs)
            return object()

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)

    class Model:
        async def ainvoke(self, _messages):
            return AIMessage(content="route bridge answer")

    runner = ProtectedChatRunner(
        lambda _request: LangGraphAgent(model=Model()), security_timeline=True
    )
    safeharness_routes.register_protected_chat_stream(runner.astream_context)

    with _test_client(_principal(role="member", can_view=True)) as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            headers={"X-Trace-ID": "trace-e2e"},
            json={
                "conversation_id": "chat-e2e",
                "agent_id": "main_agent",
                "message": "Summarize the active alarm.",
            },
        )

    assert response.status_code == 200
    payloads = _sse_payloads(response)
    assert payloads[0]["type"] == "run_started"
    assert payloads[-1]["type"] == "run_finished"
    assert any(item["type"] == "policy_decision" for item in payloads)
    assert any(item["type"] == "audit_reference" for item in payloads)
    assert any(item["data"] == {"content": "route bridge answer"} for item in payloads)
    assert all(item["trace_id"] == "trace-e2e" for item in payloads)
    assert [item["role"] for item in captured] == ["user", "assistant"]
    assert captured[-1]["extra_data"]["protected"] is True
    assert captured[-1]["extra_data"]["security_timeline"]
    assert "arguments" not in str(captured[-1]["extra_data"]["security_timeline"])


def test_protected_chat_rejects_unknown_request_field():
    async def handler(_context):
        if False:
            yield {}

    safeharness_routes.register_protected_chat_stream(handler)
    with _test_client() as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            json={
                "conversation_id": "chat-1",
                "agent_id": "main_agent",
                "message": "Summarize the alarm.",
                "tenant_id": "tenant-b",
            },
        )

    assert response.status_code == 422
    assert "detail" not in response.json()


def test_protected_chat_drops_mismatched_handler_event(monkeypatch):
    monkeypatch.setattr(safeharness_routes, "_validate_chat_scope", lambda *_args: None)

    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, _chat_id, _user_id):
            return object()

        def add_message(self, **_kwargs):
            return object()

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)

    async def handler(context):
        event = _chat_event(context, "message", 1, {"content": "secret raw result"})
        event["tenant_id"] = "tenant-b"
        yield event

    safeharness_routes.register_protected_chat_stream(handler)
    with _test_client() as client:
        response = client.post(
            "/v1/safeharness/chat/stream",
            json={
                "conversation_id": "chat-1",
                "agent_id": "main_agent",
                "message": "Summarize the alarm.",
            },
        )

    payloads = _sse_payloads(response)
    assert [item["type"] for item in payloads] == [
        "run_started",
        "error",
        "run_finished",
    ]
    assert "secret raw result" not in response.text
    assert all(item["tenant_id"] == "tenant-a" for item in payloads)


def test_protected_chat_stream_closes_handler_when_consumer_stops(monkeypatch):
    closed = False

    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, _chat_id, _user_id):
            return object()

        def add_message(self, **_kwargs):
            return object()

    async def handler(context):
        nonlocal closed
        try:
            yield _chat_event(context, "run_started", 1, {})
            await asyncio.Event().wait()
        finally:
            closed = True

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)
    safeharness_routes.register_protected_chat_stream(handler)

    async def consume_one_event():
        response = await safeharness_routes.stream_safeharness_chat(
            safeharness_routes.SafeHarnessChatRequest(
                conversation_id="chat-1",
                agent_id="main_agent",
                message="Summarize the alarm.",
            ),
            _principal(role="member", can_view=True),
            _DB(),
        )
        iterator = response.body_iterator
        await anext(iterator)
        await iterator.aclose()

    asyncio.run(consume_one_event())
    assert closed is True


def test_mutation_openapi_keeps_strict_body_contract():
    app = FastAPI()
    app.include_router(safeharness_routes.router)
    schema = app.openapi()

    prepare = schema["paths"]["/v1/safeharness/mutations"]["post"]["requestBody"]
    operation = prepare["content"]["application/json"]["schema"]
    assert prepare["required"] is True
    assert operation["additionalProperties"] is False
    assert set(operation["required"]) == {
        "schema_version", "action", "target", "reason", "idempotency_key", "reauth"
    }
    decision = schema["paths"]["/v1/safeharness/mutations/{approval_id}/decision"]["post"]["requestBody"]
    decision_schema = decision["content"]["application/json"]["schema"]
    assert decision["required"] is True
    assert decision_schema["additionalProperties"] is False
    assert set(decision_schema["required"]) == {"schema_version", "decision", "idempotency_key"}


def test_unknown_safeharness_path_is_not_proxied():
    with _test_client() as client:
        response = client.get("/v1/safeharness/not-allowlisted")

    assert response.status_code == 404


def test_policy_center_is_read_only_and_simulates_current_rules():
    with _test_client(_principal(role="member", can_view=True)) as client:
        policies = client.get("/v1/safeharness/policies")
        simulation = client.post(
            "/v1/safeharness/policies/simulate",
            json={
                "schema_version": "1.0",
                "package_id": "safeharness.mutations.v1",
                "operation": "delegation.revoke",
            },
        )
        mismatch = client.post(
            "/v1/safeharness/policies/simulate",
            json={
                "schema_version": "1.0",
                "package_id": "safeharness.default.v1",
                "operation": "delegation.revoke",
            },
        )

    assert policies.status_code == 200
    assert policies.json()["data"]["diff"]["status"] == "unavailable"
    assert policies.json()["data"]["publishing"]["available"] is False
    assert simulation.status_code == 200
    assert simulation.json()["data"]["decision"] == "require_approval"
    assert simulation.json()["data"]["impact"]["writes_control_plane"] is True
    assert mismatch.status_code == 422


def test_http_route_preserves_authentication_failure():
    with _test_client(status_code=401) as client:
        response = client.get("/v1/safeharness/overview")

    assert response.status_code == 401
    assert response.json()["data"]["error"]["code"] == "safeharness_unauthorized"
    assert "detail" not in response.json()


def test_http_route_denies_viewer_management_access():
    with _test_client(_principal(role="member", can_view=True)) as client:
        response = client.post(
            "/v1/safeharness/mutations",
            json=_mutation_body(),
        )

    assert response.status_code == 403
    assert response.json()["data"]["error"]["code"] == "safeharness_forbidden"


def test_http_route_rejects_cross_tenant_mutation():
    with _test_client() as client:
        response = client.post(
            "/v1/safeharness/mutations",
            json=_mutation_body(target={"tenant_id": "tenant-b", "agent_id": "agent-1"}),
        )

    assert response.status_code == 403


def test_http_route_rejects_unknown_mutation_field():
    with _test_client() as client:
        response = client.post(
            "/v1/safeharness/mutations",
            json={**_mutation_body(), "unexpected": True},
        )

    assert response.status_code == 422


def test_http_route_rejects_duplicate_json_fields():
    body = json.dumps(_mutation_body()).replace(
        '"action": "agent.quarantine"',
        '"action": "agent.quarantine", "action": "agent.restore"',
    ).encode()
    with _test_client() as client:
        response = client.post(
            "/v1/safeharness/mutations",
            content=body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422


def test_http_overview_returns_sumc_envelope(monkeypatch):
    called = {}

    async def upstream(method, route, **_kwargs):
        assert (method, route) == ("GET", "overview")
        called["trace_id"] = safeharness_routes.trace_id_var.get()
        return {"ok": True, "state": {"agents": 2}}

    monkeypatch.setattr(safeharness_routes, "_upstream_request", upstream)
    with _test_client() as client:
        response = client.get("/v1/safeharness/overview", headers={"X-Trace-ID": "trace-fixed"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"] == {"ok": True, "state": {"agents": 2}}
    assert payload["trace_id"] == "trace-fixed"
    assert called["trace_id"] == "trace-fixed"


def test_http_audit_forces_authoritative_tenant(monkeypatch):
    called = {}

    async def upstream(method, route, **kwargs):
        called.update(method=method, route=route, **kwargs)
        return {"ok": True, "available": True, "records": []}

    monkeypatch.setattr(safeharness_routes, "_upstream_request", upstream)
    with _test_client(_principal(role="member", can_view=True)) as client:
        response = client.get("/v1/safeharness/audit?limit=10&source=agent")

    assert response.status_code == 200
    assert called["params"] == {"tenant_id": "tenant-a", "limit": "10", "source": "agent"}


def test_http_controlled_mutation_executes_once_with_bound_envelope(monkeypatch):
    called = {}

    monkeypatch.setattr(
        "core.safeharness_mutations._verify_local_reauth",
        lambda *_args, **_kwargs: True,
    )

    async def upstream(method, selected_route, **kwargs):
        called.update(method=method, route=selected_route, **kwargs)
        return {
            "ok": True,
            "version": 2,
            "signature": "a" * 64,
            "operation": "quarantine_agent",
        }

    monkeypatch.setattr(safeharness_routes, "_upstream_request", upstream)
    with _test_client() as client:
        prepared = client.post("/v1/safeharness/mutations", json=_mutation_body())
        approval_id = prepared.json()["data"]["approval_id"]
        response = client.post(
            f"/v1/safeharness/mutations/{approval_id}/decision",
            json={
                "schema_version": "1.0",
                "decision": "allow",
                "idempotency_key": "mutation-key-0001",
            },
        )
        replay = client.post(
            f"/v1/safeharness/mutations/{approval_id}/decision",
            json={
                "schema_version": "1.0",
                "decision": "allow",
                "idempotency_key": "mutation-key-0001",
            },
        )

    assert prepared.status_code == 200
    assert response.status_code == 200
    assert replay.status_code == 200
    assert called["method"] == "POST"
    assert called["route"] == "quarantine"
    assert called["payload"]["tenant_id"] == "tenant-a"
    assert called["payload"]["agent_id"] == "agent-1"
    assert called["payload"]["operator_id"] == "user-1"
    assert called["payload"]["reason"] == "Incident containment"
    assert called["payload"]["approval_id"] == approval_id
    assert response.json()["data"]["receipt"]["gateway_verified"] is False


def test_mutation_requires_authoritative_reauthentication(monkeypatch):
    monkeypatch.setattr(
        "core.safeharness_mutations._verify_local_reauth",
        lambda *_args, **_kwargs: False,
    )
    with _test_client() as client:
        response = client.post("/v1/safeharness/mutations", json=_mutation_body())

    assert response.status_code == 403
    assert response.json()["data"]["error"]["code"] == "safeharness_reauth_failed"


def test_mutation_idempotency_key_cannot_change_target(monkeypatch):
    monkeypatch.setattr(
        "core.safeharness_mutations._verify_local_reauth",
        lambda *_args, **_kwargs: True,
    )
    with _test_client() as client:
        first = client.post("/v1/safeharness/mutations", json=_mutation_body())
        conflict = client.post(
            "/v1/safeharness/mutations",
            json=_mutation_body(target={"tenant_id": "tenant-a", "agent_id": "agent-2"}),
        )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["data"]["error"]["code"] == "safeharness_idempotency_conflict"


def test_mutation_approval_rejects_wrong_idempotency_key(monkeypatch):
    monkeypatch.setattr(
        "core.safeharness_mutations._verify_local_reauth",
        lambda *_args, **_kwargs: True,
    )
    with _test_client() as client:
        prepared = client.post("/v1/safeharness/mutations", json=_mutation_body())
        approval_id = prepared.json()["data"]["approval_id"]
        response = client.post(
            f"/v1/safeharness/mutations/{approval_id}/decision",
            json={
                "schema_version": "1.0",
                "decision": "allow",
                "idempotency_key": "mutation-key-wrong",
            },
        )

    assert response.status_code == 403
    assert response.json()["data"]["error"]["code"] == "safeharness_approval_mismatch"


def test_mutation_deny_does_not_execute(monkeypatch):
    monkeypatch.setattr("core.safeharness_mutations._verify_local_reauth", lambda *_a, **_k: True)

    async def upstream(*_args, **_kwargs):
        raise AssertionError("denied mutation must not execute")

    monkeypatch.setattr(safeharness_routes, "_upstream_request", upstream)
    with _test_client() as client:
        prepared = client.post("/v1/safeharness/mutations", json=_mutation_body())
        response = client.post(
            f"/v1/safeharness/mutations/{prepared.json()['data']['approval_id']}/decision",
            json={"schema_version": "1.0", "decision": "deny", "idempotency_key": "mutation-key-0001"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "denied"


def test_mutation_expired_approval_is_gone(monkeypatch):
    monkeypatch.setattr("core.safeharness_mutations._verify_local_reauth", lambda *_a, **_k: True)
    with _test_client() as client:
        prepared = client.post("/v1/safeharness/mutations", json=_mutation_body())
        approval_id = prepared.json()["data"]["approval_id"]
        safeharness_routes.mutation_coordinator._by_approval[approval_id].expires_at = 0
        response = client.post(
            f"/v1/safeharness/mutations/{approval_id}/decision",
            json={"schema_version": "1.0", "decision": "allow", "idempotency_key": "mutation-key-0001"},
        )

    assert response.status_code == 410


def test_mutation_upstream_failure_becomes_indeterminate(monkeypatch):
    monkeypatch.setattr("core.safeharness_mutations._verify_local_reauth", lambda *_a, **_k: True)

    async def unavailable(*_args, **_kwargs):
        raise safeharness_routes.SafeHarnessUpstreamError("unknown execution result")

    monkeypatch.setattr(safeharness_routes, "_upstream_request", unavailable)
    with _test_client() as client:
        prepared = client.post("/v1/safeharness/mutations", json=_mutation_body())
        path = f"/v1/safeharness/mutations/{prepared.json()['data']['approval_id']}/decision"
        body = {"schema_version": "1.0", "decision": "allow", "idempotency_key": "mutation-key-0001"}
        failed = client.post(path, json=body)
        replay = client.post(path, json=body)

    assert failed.status_code == 503
    assert replay.status_code == 409
    assert failed.json()["data"]["error"]["code"] == "safeharness_mutation_indeterminate"
    assert replay.json()["data"]["error"]["code"] == "safeharness_mutation_indeterminate"


def test_controlled_delegation_revoke_uses_bound_target(monkeypatch):
    monkeypatch.setattr("core.safeharness_mutations._verify_local_reauth", lambda *_a, **_k: True)
    called = {}

    async def upstream(method, route, **kwargs):
        called.update(method=method, route=route, **kwargs)
        return {"ok": True, "version": 3, "signature": "b" * 64, "operation": "revoke_delegation"}

    monkeypatch.setattr(safeharness_routes, "_upstream_request", upstream)
    body = _mutation_body(
        action="delegation.revoke",
        target={"tenant_id": "tenant-a", "delegation_id": "delegation-1"},
    )
    with _test_client() as client:
        prepared = client.post("/v1/safeharness/mutations", json=body)
        response = client.post(
            f"/v1/safeharness/mutations/{prepared.json()['data']['approval_id']}/decision",
            json={"schema_version": "1.0", "decision": "allow", "idempotency_key": "mutation-key-0001"},
        )

    assert response.status_code == 200
    assert called["route"] == "delegation_revoke"
    assert called["payload"]["delegation_id"] == "delegation-1"
    assert "agent_id" not in called["payload"]


def test_http_upstream_failure_is_minimized(monkeypatch):
    async def unavailable(*_args, **_kwargs):
        raise safeharness_routes.SafeHarnessUpstreamError("secret upstream detail")

    monkeypatch.setattr(safeharness_routes, "_upstream_request", unavailable)
    with _test_client() as client:
        response = client.get("/v1/safeharness/overview")

    assert response.status_code == 503
    assert "secret upstream detail" not in response.text
    assert response.json()["data"]["error"]["code"] == "safeharness_unavailable"
    assert "detail" not in response.json()


def test_principal_requires_explicit_tenant_and_role():
    assert (
        safeharness_routes.resolve_safeharness_principal(_DB({"role": "super_admin"}), _user())
        is None
    )
    assert (
        safeharness_routes.resolve_safeharness_principal(_DB({"tenant_id": "tenant-a"}), _user())
        is None
    )


def test_super_admin_principal_has_view_and_manage_access():
    principal = safeharness_routes.resolve_safeharness_principal(
        _DB({"tenant_id": "tenant-a", "role": "super_admin"}), _user()
    )

    assert principal is not None
    assert principal.tenant_id == "tenant-a"
    assert principal.can_view is True
    assert principal.can_manage is True


def test_read_only_security_viewer_cannot_manage():
    with pytest.raises(HTTPException) as exc:
        safeharness_routes._require_manage(_principal(role="member", can_view=True))

    assert exc.value.status_code == 403


def test_cross_tenant_operation_is_rejected():
    with pytest.raises(HTTPException) as exc:
        safeharness_routes._require_same_tenant("tenant-b", _principal())

    assert exc.value.status_code == 403


def test_mutation_operation_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        safeharness_routes.MutationPrepareRequest.model_validate(
            {**_mutation_body(), "unexpected": True}
        )


def test_mutation_operation_rejects_identifier_suffixes():
    with pytest.raises(ValidationError):
        safeharness_routes.MutationPrepareRequest.model_validate(
            _mutation_body(target={"tenant_id": "tenant-a!", "agent_id": "agent-1"})
        )


@pytest.mark.parametrize(
    "query",
    [b"tenant_id=tenant-a&unexpected=1", b"limit=10&limit=20"],
)
def test_audit_query_rejects_unknown_or_duplicate_fields(query):
    with pytest.raises(HTTPException) as exc:
        safeharness_routes._validate_audit_query(_request(query=query))

    assert exc.value.status_code == 422


@pytest.mark.parametrize(
    "query",
    [b"conversation_id=chat-1&unexpected=1", b"agent_id=agent-1&agent_id=agent-2"],
)
def test_discovery_query_rejects_unknown_or_duplicate_fields(query):
    with pytest.raises(HTTPException) as exc:
        safeharness_routes.validate_discovery_query(_request(query=query))

    assert exc.value.status_code == 422


def test_request_body_size_is_bounded():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            safeharness_routes._read_json_model(
                _request(
                    body=b"x" * 8193,
                    headers=[
                        (b"content-type", b"application/json"),
                        (b"content-length", b"8193"),
                    ],
                ),
                safeharness_routes.MutationPrepareRequest,
            )
        )

    assert exc.value.status_code == 413


def test_upstream_request_uses_admin_ui_token_and_named_path(configured_bff):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/overview"
        assert request.headers["X-SafeHarness-UI-Token"] == "u" * 32
        assert request.headers["X-Trace-ID"] == "trace-upstream"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={"ok": True, "state": {}},
        )

    token = safeharness_routes.trace_id_var.set("trace-upstream")
    try:
        result = asyncio.run(
            safeharness_routes._upstream_request(
                "GET", "overview", transport=httpx.MockTransport(handler)
            )
        )
    finally:
        safeharness_routes.trace_id_var.reset(token)

    assert result == {"ok": True, "state": {}}


def test_upstream_response_size_is_bounded(configured_bff):
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b"{" + b"x" * 1024 + b"}",
        )
    )

    with pytest.raises(safeharness_routes.SafeHarnessUpstreamError):
        asyncio.run(safeharness_routes._upstream_request("GET", "overview", transport=transport))


def test_upstream_duplicate_json_fields_fail_closed(configured_bff):
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b'{"ok":true,"ok":false}',
        )
    )

    with pytest.raises(safeharness_routes.SafeHarnessUpstreamError):
        asyncio.run(safeharness_routes._upstream_request("GET", "overview", transport=transport))


def test_upstream_timeout_fails_closed(configured_bff):
    def timeout(request: httpx.Request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(safeharness_routes.SafeHarnessUpstreamError):
        asyncio.run(
            safeharness_routes._upstream_request(
                "GET", "overview", transport=httpx.MockTransport(timeout)
            )
        )


def test_upstream_request_does_not_swallow_cancellation(configured_bff):
    def cancelled(_request: httpx.Request):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            safeharness_routes._upstream_request(
                "GET", "overview", transport=httpx.MockTransport(cancelled)
            )
        )


def test_discovery_enables_capabilities_only_after_live_probe(monkeypatch):
    async def probe(*_args, **_kwargs):
        return {"ok": True}

    monkeypatch.setattr(safeharness_routes, "_upstream_request", probe)
    payload = asyncio.run(
        safeharness_routes.capability_discovery_payload(
            _user(),
            _DB({"tenant_id": "tenant-a", "role": "super_admin"}),
            trace_id="trace-discovery",
        )
    )

    assert payload["available"] is True
    assert payload["capabilities"]["agents"] is True
    assert payload["capabilities"]["security_operations"] is True
    assert payload["capabilities"]["capability_writes"] is False
    assert payload["capabilities"]["application_execution"] is False
    assert payload["context"]["tenant_id"] == "tenant-a"
    assert payload["context"]["trace_id"] == "trace-discovery"


def test_discovery_exposes_protected_chat_only_when_handler_registered(monkeypatch):
    async def probe(*_args, **_kwargs):
        return {"ok": True}

    async def handler(_context):
        if False:
            yield {}

    monkeypatch.setattr(safeharness_routes, "_upstream_request", probe)
    safeharness_routes.register_protected_chat_stream(handler)
    payload = asyncio.run(
        safeharness_routes.capability_discovery_payload(
            _user(),
            _DB({"tenant_id": "tenant-a", "role": "super_admin"}),
        )
    )

    assert payload["capabilities"]["protected_chat"] is True


def test_discovery_rejects_cross_user_conversation(monkeypatch):
    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, chat_id, user_id):
            assert (chat_id, user_id) == ("chat-other", "user-1")
            return None

    async def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("upstream must not be probed")

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)
    monkeypatch.setattr(safeharness_routes, "_upstream_request", unexpected_probe)
    payload = asyncio.run(
        safeharness_routes.capability_discovery_payload(
            _user(),
            _DB({"tenant_id": "tenant-a", "role": "super_admin"}),
            conversation_id="chat-other",
        )
    )

    assert payload["available"] is False
    assert payload["context"]["conversation_id"] is None


def test_discovery_binds_authorized_project_context(monkeypatch):
    class ChatService:
        def __init__(self, _db):
            pass

        def get_session_with_access(self, chat_id, user_id):
            assert (chat_id, user_id) == ("chat-project", "user-1")
            return SimpleNamespace(project_id="project-1"), "admin"

    class ProjectService:
        def __init__(self, _db):
            pass

        def get_raw(self, project_id):
            assert project_id == "project-1"
            return SimpleNamespace(
                project_id=project_id,
                deleted_at=None,
                kind="personal",
                owner_user_id="user-1",
            )

    async def probe(*_args, **_kwargs):
        return {"ok": True}

    monkeypatch.setattr(safeharness_routes, "ChatService", ChatService)
    monkeypatch.setattr(safeharness_routes, "ProjectService", ProjectService)
    monkeypatch.setattr(safeharness_routes, "_upstream_request", probe)
    payload = asyncio.run(
        safeharness_routes.capability_discovery_payload(
            _user(),
            _DB({"tenant_id": "tenant-a", "role": "super_admin"}),
            conversation_id="chat-project",
            project_id="project-1",
        )
    )

    assert payload["available"] is True
    assert payload["context"]["conversation_id"] == "chat-project"
    assert payload["context"]["project_id"] == "project-1"


def test_discovery_rejects_invisible_agent(monkeypatch):
    class UserAgentService:
        def __init__(self, _db):
            pass

        def get_by_id(self, agent_id, user_id=None):
            assert (agent_id, user_id) == ("agent-other", "user-1")
            raise PermissionError("not visible")

    async def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("upstream must not be probed")

    monkeypatch.setattr(safeharness_routes, "UserAgentService", UserAgentService)
    monkeypatch.setattr(safeharness_routes, "_upstream_request", unexpected_probe)
    payload = asyncio.run(
        safeharness_routes.capability_discovery_payload(
            _user(),
            _DB({"tenant_id": "tenant-a", "role": "super_admin"}),
            agent_id="agent-other",
        )
    )

    assert payload["available"] is False
    assert payload["context"]["agent_id"] is None
