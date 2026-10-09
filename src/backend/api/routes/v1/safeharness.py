"""Fail-closed, allowlisted BFF for SafeHarness management operations."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path as FilePath
import re
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
from core.auth.backend import UserContext, require_auth
from core.auth.capabilities import user_has_capability
from core.auth.permissions_iface import resolve_project_permission
from core.config.settings import settings
from core.db.engine import get_db
from core.db.models import UserShadow
from core.infra.logging import trace_id_var
from core.infra.exceptions import BadRequestError, ResourceNotFoundError
from core.infra.responses import (
    error_response,
    generate_trace_id,
    sse_response,
    success_response,
)
from core.services import ChatService, UserAgentService
from core.services.user_model_selection import UserModelSelectionError, resolve_user_model_provider_id
from core.services.project_service import ProjectService
from core.safeharness_mutations import (
    APPROVAL_TTL_SECONDS,
    MutationConflict,
    MutationDecisionRequest,
    MutationIndeterminate,
    MutationPrepareRequest,
    MutationUnavailable,
    _decision_payload,
    mutation_coordinator,
)
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$"
_IDENTIFIER = re.compile(_IDENTIFIER_PATTERN)
_MAX_REQUEST_BYTES = 8 * 1024
_MAIN_AGENT_ID = "main_agent"
_ALLOWED_AUDIT_QUERY = frozenset({"tenant_id", "source", "action", "limit"})
_ALLOWED_DISCOVERY_QUERY = frozenset({"conversation_id", "agent_id", "project_id"})
_SECURITY_REPORT_SCENARIOS = frozenset(
    {"input-injection", "nested-tool-result", "restored-tool-call", "restored-history"}
)
_SECURITY_REPORT_PUBLIC_FIELDS = {
    "input-injection": ("input_guard", "用户注入在 input_guard 被拒绝，模型和工具均不运行。"),
    "nested-tool-result": ("post_tool_guard", "工具执行后返回嵌套注入；结果被拒绝，原始恶意文本不进入模型。"),
    "restored-tool-call": ("restored_message_guard", "恢复的 Assistant 消息携带恶意工具调用；调用元数据被清除。"),
    "restored-history": ("restored_history_guard", "恢复的历史消息被重新检查和替换，后续模型调用被跳过。"),
}
_SECURITY_REPORT_EVIDENCE_FIELDS = frozenset(
    {
        "blocked",
        "model_requests",
        "model_calls",
        "tool_calls_started",
        "tool_executor_calls",
        "blocked_tool_results",
        "blocked_status_seen_by_model",
        "unsafe_result_seen_by_model",
        "tool_calls_stripped",
        "tool_calls_proposed",
        "unsafe_history_retained",
    }
)
_UPSTREAM_ROUTES = {
    "overview": "/api/overview",
    "audit": "/api/audit",
    "quarantine": "/api/agents/quarantine",
    "restore": "/api/agents/restore",
    "delegation_revoke": "/api/delegations/revoke",
    "application_authorize": "/api/application-mutations/authorize",
}
_POLICY_PACKAGES = (
    {
        "id": "safeharness.default.v1",
        "status": "active",
        "operations": ["internet_search"],
        "enforcement": ["capability", "one_shot_approval", "pre_tool", "post_tool"],
        "production_ready": False,
    },
    {
        "id": "safeharness.mutations.v1",
        "status": "active",
        "operations": [
            "agent.quarantine",
            "agent.restore",
            "delegation.revoke",
            "capability.enable",
            "capability.disable",
            "plugin.enable",
            "plugin.disable",
            "automation.run",
        ],
        "enforcement": ["super_admin", "tenant_binding", "password_reauth", "one_shot_approval", "idempotency"],
        "production_ready": False,
    },
)
_UI_CAPABILITIES = {
    "chat": True,
    "agents": True,
    "kb": True,
    "app_center": True,
    "projects": True,
    "project_detail": True,
    "my_space": True,
    "settings": True,
    "ability_center": True,
    "skills": True,
    "mcp": True,
    "lab": False,
    "docs": False,
    "share_records": False,
    "protected_chat": False,
    "security_operations": True,
    # Inventory stays readable while capability mutations and application
    # execution lack SafeHarness policy, approval, and Gateway enforcement.
    "capability_writes": False,
    "application_execution": False,
}

_CHAT_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "type",
        "sequence",
        "tenant_id",
        "user_id",
        "agent_id",
        "conversation_id",
        "session_id",
        "run_id",
        "trace_id",
        "timestamp",
        "data",
    }
)
_CHAT_EVENT_DATA_FIELDS = {
    "run_started": frozenset(),
    "policy_decision": frozenset({"decision_id", "event_kind", "disposition", "policy_version", "reason_code"}),
    "tool_proposal": frozenset({"decision_id", "tool_name"}),
    "approval": frozenset({"approval_id", "status"}),
    "execution": frozenset({"decision_id", "tool_name", "status"}),
    "receipt": frozenset({"receipt_id", "decision_id", "enforced", "adapter"}),
    "audit_reference": frozenset({"record_type", "event_id", "decision_id", "receipt_id"}),
    "message": frozenset({"content"}),
    "blocked": frozenset({"reason_code", "message"}),
    "error": frozenset({"code", "message"}),
    "run_finished": frozenset({"status"}),
}
_JUDGE_EVENT_FIELDS = frozenset({"judge_status", "judge_calls", "judge_model"})
_CHAT_EVENT_OPTIONAL_DATA_FIELDS = {
    "policy_decision": frozenset({"stage", "route", "agent_id"}) | _JUDGE_EVENT_FIELDS,
    "execution": frozenset({"stage", "route", "agent_id"}),
    "blocked": frozenset(
        {
            "source",
            "action",
            "route",
            "agent_chain",
            "model_calls",
            "tool_calls_proposed",
            "tool_calls_started",
            "final_status",
        }
    ) | _JUDGE_EVENT_FIELDS,
    "error": _JUDGE_EVENT_FIELDS,
    "run_finished": frozenset(
        {
            "route",
            "agent_id",
            "agent_chain",
            "model_calls",
            "tool_calls_proposed",
            "tool_calls_started",
        }
    ) | _JUDGE_EVENT_FIELDS,
}

ProtectedChatStreamHandler = Callable[[Mapping[str, str]], AsyncIterator[Mapping[str, object]]]
_protected_chat_stream_handler: Optional[ProtectedChatStreamHandler] = None


@dataclass(frozen=True)
class SafeHarnessPrincipal:
    user_id: str
    tenant_id: str
    role: str
    can_view: bool
    can_manage: bool


class SafeHarnessChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    conversation_id: str = Field(min_length=1, max_length=100, pattern=_IDENTIFIER_PATTERN)
    agent_id: str = Field(min_length=1, max_length=256, pattern=_IDENTIFIER_PATTERN)
    project_id: Optional[str] = Field(default=None, min_length=1, max_length=256, pattern=_IDENTIFIER_PATTERN)
    message: str = Field(min_length=1, max_length=10_000)
    model_provider_id: Optional[str] = Field(default=None, min_length=1, max_length=100, pattern=_IDENTIFIER_PATTERN)


_MUTATION_PREPARE_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {"application/json": {"schema": MutationPrepareRequest.model_json_schema()}},
    }
}
_MUTATION_DECISION_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {"application/json": {"schema": MutationDecisionRequest.model_json_schema()}},
    }
}


class PolicySimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"]
    package_id: Literal["safeharness.default.v1", "safeharness.mutations.v1"]
    operation: Literal[
        "internet_search", "agent.quarantine", "agent.restore", "delegation.revoke"
    ]


_POLICY_SIMULATION_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {"application/json": {"schema": PolicySimulationRequest.model_json_schema()}},
    }
}


class SafeHarnessUpstreamError(RuntimeError):
    pass


def register_protected_chat_stream(
    handler: ProtectedChatStreamHandler, *, replace: bool = False
) -> None:
    """Register optional SafeHarness LangGraph stream adapter.

    SUMC does not own the protected graph.  A deployment bootstrapper may
    register an adapter when the SafeHarness package and LangGraph backend are
    installed; absent registration, the route remains unavailable.
    """

    global _protected_chat_stream_handler
    if not callable(handler):
        raise TypeError("protected chat stream handler must be callable")
    if _protected_chat_stream_handler is not None and not replace:
        raise RuntimeError("protected chat stream handler is already registered")
    _protected_chat_stream_handler = handler


def unregister_protected_chat_stream() -> None:
    global _protected_chat_stream_handler
    _protected_chat_stream_handler = None


def protected_chat_available() -> bool:
    return _protected_chat_stream_handler is not None


def _current_trace_id() -> str:
    return trace_id_var.get() or generate_trace_id()


def safeharness_error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: Optional[dict[str, str]] = None,
):
    response = error_response(
        code=status_code,
        message=message,
        data={"error": {"code": code}},
        status_code=status_code,
        trace_id=_current_trace_id(),
    )
    if headers:
        response.headers.update(headers)
    return response


class SafeHarnessRoute(APIRoute):
    """Keep every SafeHarness BFF failure in the normal API envelope."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            token = None
            if not trace_id_var.get():
                trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()
                token = trace_id_var.set(trace_id)
            try:
                return await original(request)
            except RequestValidationError:
                return safeharness_error_response(
                    422,
                    "safeharness_invalid_request",
                    "SafeHarness request validation failed.",
                )
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                code = detail.get("code")
                message = detail.get("message")
                if not isinstance(code, str):
                    code = (
                        "safeharness_unauthorized"
                        if exc.status_code == 401
                        else "safeharness_request_failed"
                    )
                if not isinstance(message, str):
                    message = (
                        "SafeHarness authentication is required."
                        if exc.status_code == 401
                        else "SafeHarness request failed."
                    )
                return safeharness_error_response(
                    exc.status_code,
                    code,
                    message,
                    headers=exc.headers,
                )
            finally:
                if token is not None:
                    trace_id_var.reset(token)

        return handler


router = APIRouter(
    prefix="/v1/safeharness",
    tags=["safeharness"],
    route_class=SafeHarnessRoute,
)


def _strict_json_loads(raw: bytes):
    def object_without_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(_value):
        raise ValueError("non-finite JSON number")

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_without_duplicates,
        parse_constant=reject_constant,
    )


def resolve_safeharness_principal(
    db: Session,
    user: UserContext,
) -> Optional[SafeHarnessPrincipal]:
    """Resolve only explicit tenant/role metadata; never guess an authority scope."""
    try:
        shadow = db.get(UserShadow, str(user.user_id))
        metadata = shadow.extra_data if shadow and isinstance(shadow.extra_data, dict) else {}
        tenant_id = metadata.get("tenant_id")
        role = metadata.get("role")
        if not isinstance(tenant_id, str) or _IDENTIFIER.fullmatch(tenant_id.strip()) is None:
            return None
        if not isinstance(role, str) or _IDENTIFIER.fullmatch(role.strip()) is None:
            return None
        tenant_id = tenant_id.strip()
        role = role.strip()
        can_manage = role == "super_admin"
        can_view = can_manage or user_has_capability(db, str(user.user_id), "can_security_view")
        return SafeHarnessPrincipal(
            user_id=str(user.user_id),
            tenant_id=tenant_id,
            role=role,
            can_view=can_view,
            can_manage=can_manage,
        )
    except Exception:  # noqa: BLE001 - identity resolution must fail closed
        return None


def _principal_dependency(
    user: UserContext = Depends(require_auth(True)),
    db: Session = Depends(get_db),
) -> SafeHarnessPrincipal:
    principal = resolve_safeharness_principal(db, user)
    if principal is None:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "safeharness_context_unavailable",
                "message": "Authoritative SafeHarness identity context is unavailable.",
            },
        )
    return principal


def _require_view(
    principal: SafeHarnessPrincipal = Depends(_principal_dependency),
) -> SafeHarnessPrincipal:
    if not principal.can_view:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "safeharness_forbidden",
                "message": "SafeHarness audit access is not authorized.",
            },
        )
    return principal


def _require_manage(
    principal: SafeHarnessPrincipal = Depends(_principal_dependency),
) -> SafeHarnessPrincipal:
    if not principal.can_manage:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "safeharness_forbidden",
                "message": "SafeHarness management access requires super_admin.",
            },
        )
    return principal


def _admin_ui_base_url() -> str:
    raw = settings.safeharness.admin_ui_url
    token = settings.safeharness.ui_access_token
    timeout = settings.safeharness.timeout_seconds
    max_bytes = settings.safeharness.max_response_bytes
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or len(token) < 32
        or not 1 <= timeout <= 30
        or not 1024 <= max_bytes <= 4 * 1024 * 1024
    ):
        raise SafeHarnessUpstreamError("SafeHarness BFF is not configured safely")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


async def _upstream_request(
    method: Literal["GET", "POST"],
    route: Literal["overview", "audit", "quarantine", "restore", "delegation_revoke", "application_authorize"],
    *,
    params: Optional[dict[str, str]] = None,
    payload: Optional[dict[str, str]] = None,
    trace_id: Optional[str] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> dict:
    """Call one named Admin UI operation and bound the returned JSON object."""
    base_url = _admin_ui_base_url()
    headers = {
        "Accept": "application/json",
        "X-SafeHarness-UI-Token": settings.safeharness.ui_access_token,
        "X-Trace-ID": trace_id or _current_trace_id(),
    }
    if method == "POST":
        mutation_key = settings.safeharness.mutation_key
        if len(mutation_key) < 32:
            raise SafeHarnessUpstreamError("SafeHarness mutation channel is not configured")
        headers["X-SafeHarness-Mutation-Key"] = mutation_key
    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=settings.safeharness.timeout_seconds,
            transport=transport,
        ) as client:
            async with client.stream(
                method,
                base_url + _UPSTREAM_ROUTES[route],
                headers=headers,
                params=params,
                json=payload,
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise SafeHarnessUpstreamError("SafeHarness upstream rejected the request")
                if "application/json" not in response.headers.get("content-type", "").lower():
                    raise SafeHarnessUpstreamError("SafeHarness upstream returned non-JSON data")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > settings.safeharness.max_response_bytes:
                        raise SafeHarnessUpstreamError("SafeHarness upstream response is too large")
    except SafeHarnessUpstreamError:
        raise
    except httpx.TimeoutException as exc:
        raise SafeHarnessUpstreamError("SafeHarness upstream timed out") from exc
    except httpx.RequestError as exc:
        raise SafeHarnessUpstreamError("SafeHarness upstream is unavailable") from exc

    try:
        value = _strict_json_loads(bytes(body))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SafeHarnessUpstreamError("SafeHarness upstream returned invalid JSON") from exc
    if not isinstance(value, dict) or value.get("ok") is False:
        raise SafeHarnessUpstreamError("SafeHarness upstream returned an invalid response")
    return value


def _upstream_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "code": "safeharness_unavailable",
            "message": "SafeHarness control plane is unavailable.",
        },
    )


def _validate_audit_query(request: Request) -> None:
    items = list(request.query_params.multi_items())
    keys = [key for key, _ in items]
    if set(keys) - _ALLOWED_AUDIT_QUERY or len(keys) != len(set(keys)):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "safeharness_invalid_query",
                "message": "SafeHarness audit query contains unknown or duplicate fields.",
            },
        )


def validate_discovery_query(request: Request) -> None:
    items = list(request.query_params.multi_items())
    keys = [key for key, _ in items]
    if set(keys) - _ALLOWED_DISCOVERY_QUERY or len(keys) != len(set(keys)):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "safeharness_invalid_query",
                "message": "SafeHarness discovery query contains unknown or duplicate fields.",
            },
        )
    if any(_IDENTIFIER.fullmatch(value.strip()) is None for _, value in items):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "safeharness_invalid_query",
                "message": "SafeHarness discovery query contains an invalid identifier.",
            },
        )


def _require_same_tenant(requested_tenant: str, principal: SafeHarnessPrincipal) -> None:
    if requested_tenant != principal.tenant_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "safeharness_cross_tenant",
                "message": "Cross-tenant SafeHarness access is forbidden.",
            },
        )


def _validate_chat_scope(
    db: Session,
    principal: SafeHarnessPrincipal,
    operation: SafeHarnessChatRequest,
) -> None:
    try:
        session_access = ChatService(db).get_session_with_access(
            operation.conversation_id, principal.user_id
        )
        if session_access is None:
            raise LookupError("conversation not found")
        session = session_access[0] if isinstance(session_access, tuple) else session_access
        if getattr(session, "project_id", None) != operation.project_id:
            raise LookupError("conversation project mismatch")
        if operation.project_id:
            project = ProjectService(db).get_raw(operation.project_id)
            if resolve_project_permission(db, principal.user_id, project) == "none":
                raise LookupError("project not found")
        if operation.agent_id != _MAIN_AGENT_ID:
            UserAgentService(db).get_by_id(operation.agent_id, user_id=principal.user_id)
    except Exception as exc:  # noqa: BLE001 - do not expose object existence
        raise HTTPException(
            status_code=404,
            detail={
                "code": "safeharness_chat_context_not_found",
                "message": "Protected chat context is unavailable.",
            },
        ) from exc


def _normalize_chat_event(
    value: Mapping[str, object],
    context: Mapping[str, str],
    sequence: int,
) -> dict:
    if not isinstance(value, Mapping) or set(value) != _CHAT_EVENT_FIELDS:
        raise ValueError("protected chat event fields are invalid")
    event_type = value.get("type")
    if event_type not in _CHAT_EVENT_DATA_FIELDS:
        raise ValueError("protected chat event type is invalid")
    if value.get("schema_version") != "1.0" or value.get("sequence") != sequence:
        raise ValueError("protected chat event sequence is invalid")
    for name in (
        "tenant_id",
        "user_id",
        "agent_id",
        "conversation_id",
        "session_id",
        "run_id",
        "trace_id",
    ):
        if value.get(name) != context[name]:
            raise ValueError("protected chat event context is invalid")
    for name in ("event_id", "timestamp"):
        item = value.get(name)
        if not isinstance(item, str) or not item or len(item) > 128:
            raise ValueError("protected chat event metadata is invalid")
    data = value.get("data")
    required_data = _CHAT_EVENT_DATA_FIELDS[event_type]
    allowed_data = required_data | _CHAT_EVENT_OPTIONAL_DATA_FIELDS.get(event_type, frozenset())
    if not isinstance(data, Mapping) or not required_data <= set(data) or not set(data) <= allowed_data:
        raise ValueError("protected chat event data is invalid")
    normalized_data = dict(data)
    for name, item_value in normalized_data.items():
        if name in required_data:
            continue
        if name in {"model_calls", "tool_calls_proposed", "tool_calls_started", "judge_calls"}:
            if not isinstance(item_value, int) or isinstance(item_value, bool) or not 0 <= item_value <= 100_000:
                raise ValueError("protected chat evidence counter is invalid")
        elif not isinstance(item_value, str) or not item_value or len(item_value) > 512:
            raise ValueError("protected chat event detail is invalid")
    if event_type == "policy_decision":
        if any(
            not isinstance(normalized_data[name], str) or len(normalized_data[name]) > 256
            for name in ("decision_id", "event_kind", "disposition", "policy_version", "reason_code")
        ):
            raise ValueError("protected chat policy decision is invalid")
    elif event_type == "tool_proposal":
        if any(
            not isinstance(normalized_data[name], str) or len(normalized_data[name]) > 256
            for name in ("decision_id", "tool_name")
        ):
            raise ValueError("protected chat tool proposal is invalid")
    elif event_type == "approval":
        if not isinstance(normalized_data["approval_id"], str) or len(normalized_data["approval_id"]) > 256 or normalized_data["status"] not in {"approved", "denied", "not_required"}:
            raise ValueError("protected chat approval is invalid")
    elif event_type == "execution":
        if any(
            not isinstance(normalized_data[name], str) or len(normalized_data[name]) > 256
            for name in ("decision_id", "tool_name")
        ) or normalized_data["status"] not in {"started", "dispatched", "succeeded", "failed"}:
            raise ValueError("protected chat execution is invalid")
    elif event_type == "receipt":
        if any(
            not isinstance(normalized_data[name], str) or len(normalized_data[name]) > 256
            for name in ("receipt_id", "decision_id", "adapter")
        ) or not isinstance(normalized_data["enforced"], bool):
            raise ValueError("protected chat receipt is invalid")
    elif event_type == "audit_reference":
        if any(
            not isinstance(normalized_data[name], str) or len(normalized_data[name]) > 256
            for name in ("record_type", "event_id", "decision_id", "receipt_id")
        ):
            raise ValueError("protected chat audit reference is invalid")
    elif event_type == "message":
        content = normalized_data.get("content")
        if not isinstance(content, str) or len(content) > 100_000:
            raise ValueError("protected chat message is invalid")
    elif event_type == "blocked":
        reason = normalized_data.get("reason_code")
        message = normalized_data.get("message")
        if (
            not isinstance(reason, str)
            or _IDENTIFIER.fullmatch(reason) is None
            or not isinstance(message, str)
            or not message
            or len(message) > 512
        ):
            raise ValueError("protected chat block reason is invalid")
    elif event_type == "error":
        code = normalized_data.get("code")
        message = normalized_data.get("message")
        if (
            not isinstance(code, str)
            or _IDENTIFIER.fullmatch(code) is None
            or not isinstance(message, str)
            or not message
            or len(message) > 512
        ):
            raise ValueError("protected chat error is invalid")
    elif event_type == "run_finished" and normalized_data.get("status") not in {
        "completed",
        "blocked",
        "error",
    }:
        raise ValueError("protected chat terminal status is invalid")
    return {
        **{name: value[name] for name in _CHAT_EVENT_FIELDS if name != "data"},
        "data": normalized_data,
    }


def _chat_error_event(
    context: Mapping[str, str], sequence: int, event_type: Literal["error", "run_finished"]
) -> dict:
    return {
        "schema_version": "1.0",
        "event_id": str(uuid.uuid4()),
        "type": event_type,
        "sequence": sequence,
        "tenant_id": context["tenant_id"],
        "user_id": context["user_id"],
        "agent_id": context["agent_id"],
        "conversation_id": context["conversation_id"],
        "session_id": context["session_id"],
        "run_id": context["run_id"],
        "trace_id": context["trace_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": (
            {
                "code": "protected_run_failed",
                "message": "Protected chat run failed.",
            }
            if event_type == "error"
            else {"status": "error"}
        ),
    }


def _sse_chat_event(event: Mapping[str, object]) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n\n"


async def _read_json_model(request: Request, model):
    """Read one small JSON body without letting FastAPI buffer an unbounded body."""
    if request.headers.get("transfer-encoding"):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "safeharness_invalid_request",
                "message": "Chunked SafeHarness requests are not accepted.",
            },
        )
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail={
                "code": "safeharness_content_type_required",
                "message": "Content-Type must be application/json.",
            },
        )
    try:
        content_length = int(request.headers.get("content-length", ""))
    except ValueError:
        content_length = -1
    if content_length < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "safeharness_invalid_request",
                "message": "A valid Content-Length is required.",
            },
        )
    if content_length > _MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "safeharness_request_too_large",
                "message": "SafeHarness request body is too large.",
            },
        )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "safeharness_request_too_large",
                    "message": "SafeHarness request body is too large.",
                },
            )
    if len(body) != content_length:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "safeharness_invalid_request",
                "message": "SafeHarness request body length is invalid.",
            },
        )
    try:
        value = _strict_json_loads(bytes(body))
        return model.model_validate(value)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "safeharness_invalid_request",
                "message": "SafeHarness request body is invalid.",
            },
        ) from exc


async def capability_discovery_payload(
    user: Optional[UserContext],
    db: Session,
    *,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    project_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> dict:
    principal = resolve_safeharness_principal(db, user) if user is not None else None
    current_trace_id = trace_id or _current_trace_id()
    context_authorized = principal is not None
    session = None
    if context_authorized and conversation_id is not None:
        try:
            session_access = ChatService(db).get_session_with_access(conversation_id, principal.user_id)
            session = session_access[0] if isinstance(session_access, tuple) else session_access
            context_authorized = session is not None
        except Exception:  # noqa: BLE001 - discovery must fail closed
            context_authorized = False
    if context_authorized and project_id is not None:
        try:
            project = ProjectService(db).get_raw(project_id)
            context_authorized = resolve_project_permission(db, principal.user_id, project) != "none"
            if context_authorized and conversation_id is not None:
                context_authorized = getattr(session, "project_id", None) == project_id
        except Exception:  # noqa: BLE001 - discovery must fail closed
            context_authorized = False
    if context_authorized and agent_id is not None:
        try:
            if agent_id != _MAIN_AGENT_ID:
                UserAgentService(db).get_by_id(agent_id, user_id=principal.user_id)
        except (LookupError, PermissionError):
            context_authorized = False
        except Exception:  # noqa: BLE001 - discovery must fail closed
            context_authorized = False
    context = {
        "tenant_id": principal.tenant_id if principal else None,
        "user_id": str(user.user_id) if user else None,
        "role": principal.role if principal else None,
        "conversation_id": conversation_id if context_authorized else None,
        "agent_id": agent_id if context_authorized else None,
        "project_id": project_id if context_authorized else None,
        "trace_id": current_trace_id,
    }
    available = False
    if principal is not None and principal.can_view and context_authorized:
        try:
            await _upstream_request("GET", "overview", trace_id=current_trace_id)
            available = True
        except SafeHarnessUpstreamError:
            pass
    capabilities = dict(_UI_CAPABILITIES) if available else {}
    if available:
        capabilities["protected_chat"] = protected_chat_available()
    return {
        "available": available,
        "source": "backend",
        "capabilities": capabilities,
        "context": context,
    }


@router.post("/chat/stream", summary="SafeHarness protected chat stream")
async def stream_safeharness_chat(
    operation: SafeHarnessChatRequest,
    principal: SafeHarnessPrincipal = Depends(_principal_dependency),
    db: Session = Depends(get_db),
):
    handler = _protected_chat_stream_handler
    if handler is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "safeharness_chat_unavailable",
                "message": "Protected chat backend is unavailable.",
            },
        )
    chat_service = ChatService(db)
    if chat_service.get_session_with_access(operation.conversation_id, principal.user_id) is None:
        try:
            if operation.project_id:
                project = ProjectService(db).get_raw(operation.project_id)
                if resolve_project_permission(db, principal.user_id, project) == "none":
                    raise LookupError("project not found")
            if operation.agent_id != _MAIN_AGENT_ID:
                UserAgentService(db).get_by_id(operation.agent_id, user_id=principal.user_id)
            created = chat_service.ensure_session(
                chat_id=operation.conversation_id,
                user_id=principal.user_id,
                title=operation.message[:80],
                extra_data={
                    "chat_id": operation.conversation_id,
                    "safeharness_protected": True,
                    **(
                        {"agent_id": operation.agent_id}
                        if operation.agent_id != _MAIN_AGENT_ID
                        else {}
                    ),
                },
                project_id=operation.project_id,
            )
            if created is None:
                raise LookupError("conversation ownership mismatch")
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "safeharness_chat_context_not_found",
                    "message": "Protected chat context is unavailable.",
                },
            ) from exc
    _validate_chat_scope(db, principal, operation)
    update_session = getattr(chat_service, "update_session", None)
    if callable(update_session):
        update_session(
            operation.conversation_id,
            principal.user_id,
            {"extra_data": {"safeharness_protected": True}},
        )
    selected_provider_id = None
    try:
        selected_provider_id = resolve_user_model_provider_id(
            db, operation.model_provider_id, user_id=principal.user_id
        )
    except UserModelSelectionError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "safeharness_model_not_allowed", "message": str(exc)},
        ) from exc
    trace_id = _current_trace_id()
    context = {
        "tenant_id": principal.tenant_id,
        "user_id": principal.user_id,
        "role": principal.role,
        "conversation_id": operation.conversation_id,
        "session_id": str(uuid.uuid4()),
        "agent_id": operation.agent_id,
        "project_id": operation.project_id or "",
        "message": operation.message,
        "model_provider_id": selected_provider_id or "",
        "run_id": str(uuid.uuid4()),
        "delegation_id": str(uuid.uuid4()),
        "trace_id": trace_id,
    }
    stream = handler(context)
    if not callable(getattr(stream, "__aiter__", None)):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "safeharness_chat_unavailable",
                "message": "Protected chat backend is unavailable.",
            },
        )
    try:
        history_metadata = {
            "trace_id": trace_id,
            "run_id": context["run_id"],
            **({"project_id": operation.project_id} if operation.project_id else {}),
        }
        chat_service.add_message(
            chat_id=operation.conversation_id,
            role="user",
            content=operation.message,
            model="safeharness",
            extra_data=history_metadata,
        )
    except Exception as exc:  # noqa: BLE001 - persistence failure is fail closed
        raise HTTPException(
            status_code=503,
            detail={
                "code": "safeharness_chat_persistence_failed",
                "message": "Protected chat history is unavailable.",
            },
        ) from exc

    async def events():
        sequence = 0
        started = False
        assistant_response: str | None = None
        security_timeline: list[dict[str, object]] = []
        try:
            async for raw_event in stream:
                normalized = _normalize_chat_event(raw_event, context, sequence + 1)
                event_type = normalized["type"]
                if sequence == 0 and event_type != "run_started":
                    raise ValueError("protected chat stream must start with run_started")
                if sequence > 0 and event_type == "run_started":
                    raise ValueError("protected chat stream repeated run_started")
                sequence += 1
                started = True
                if event_type == "message":
                    assistant_response = str(normalized["data"]["content"])
                elif event_type == "blocked":
                    assistant_response = str(normalized["data"]["message"])
                if event_type in {
                    "policy_decision",
                    "tool_proposal",
                    "approval",
                    "execution",
                    "receipt",
                    "audit_reference",
                    "blocked",
                    "error",
                    "run_finished",
                }:
                    security_timeline.append(
                        {
                            "type": event_type,
                            "sequence": normalized["sequence"],
                            "data": dict(normalized["data"]),
                        }
                    )
                yield _sse_chat_event(normalized)
                if event_type == "run_finished":
                    if assistant_response is not None and normalized["data"]["status"] in {
                        "completed",
                        "blocked",
                    }:
                        chat_service.add_message(
                            chat_id=operation.conversation_id,
                            role="assistant",
                            content=assistant_response,
                            model="safeharness",
                            extra_data={
                                **history_metadata,
                                "protected": True,
                                "blocked": normalized["data"]["status"] == "blocked",
                                "security_timeline": security_timeline,
                            },
                        )
                    return
            raise ValueError("protected chat stream ended without run_finished")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - stream failures stay generic and fail closed
            if not started:
                sequence += 1
                yield _sse_chat_event(
                    {
                        **_chat_error_event(context, sequence, "run_finished"),
                        "type": "run_started",
                        "data": {},
                    }
                )
            sequence += 1
            yield _sse_chat_event(_chat_error_event(context, sequence, "error"))
            sequence += 1
            yield _sse_chat_event(_chat_error_event(context, sequence, "run_finished"))
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - closing must not leak backend details
                    pass

    return sse_response(events(), extra_headers={"X-Trace-ID": trace_id})


@router.get("/overview", summary="SafeHarness management overview")
async def get_safeharness_overview(
    _principal: SafeHarnessPrincipal = Depends(_require_manage),
):
    try:
        return success_response(
            data=await _upstream_request("GET", "overview"), trace_id=_current_trace_id()
        )
    except SafeHarnessUpstreamError as exc:
        raise _upstream_unavailable() from exc


@router.get("/security-report", summary="SafeHarness product demo security report")
def get_safeharness_security_report(
    _principal: SafeHarnessPrincipal = Depends(_require_view),
):
    report_path = os.getenv("SUMC_SECURITY_REPORT_PATH", "").strip()
    if not report_path:
        raise HTTPException(status_code=404, detail="Security report is unavailable.")
    try:
        report_file = FilePath(report_path)
        stat = report_file.stat()
        if stat.st_size > 64 * 1024:
            raise ValueError("report is too large")
        payload = _strict_json_loads(report_file.read_bytes())
        generated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Security report is unavailable.") from exc
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("overall_status") not in {"pass", "fail"}
        or payload.get("production_ready") is not False
        or not isinstance(scenarios, list)
        or len(scenarios) != len(_SECURITY_REPORT_SCENARIOS)
    ):
        raise HTTPException(status_code=503, detail="Security report is invalid.")
    sanitized_scenarios = []
    seen_names = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {
            "name",
            "blocked_stage",
            "description",
            "passed",
            "evidence",
        }:
            raise HTTPException(status_code=503, detail="Security report is invalid.")
        name = scenario["name"]
        blocked_stage = scenario["blocked_stage"]
        description = scenario["description"]
        evidence = scenario["evidence"]
        if (
            name not in _SECURITY_REPORT_SCENARIOS
            or name in seen_names
            or blocked_stage != _SECURITY_REPORT_PUBLIC_FIELDS[name][0]
            or description != _SECURITY_REPORT_PUBLIC_FIELDS[name][1]
            or not isinstance(description, str)
            or not 1 <= len(description) <= 300
            or not isinstance(scenario["passed"], bool)
            or not isinstance(evidence, dict)
            or not evidence
            or len(evidence) > 8
            or set(evidence) - _SECURITY_REPORT_EVIDENCE_FIELDS
            or any(type(value) not in {bool, int} for value in evidence.values())
        ):
            raise HTTPException(status_code=503, detail="Security report is invalid.")
        seen_names.add(name)
        sanitized_scenarios.append(
            {
                "name": name,
                "blocked_stage": blocked_stage,
                "description": _SECURITY_REPORT_PUBLIC_FIELDS[name][1],
                "passed": scenario["passed"],
                "evidence": evidence,
            }
        )
    if seen_names != _SECURITY_REPORT_SCENARIOS:
        raise HTTPException(status_code=503, detail="Security report is invalid.")
    return success_response(
        data={
            "schema_version": "1.0",
            "generated_at": generated_at,
            "overall_status": payload["overall_status"],
            "production_ready": False,
            "scenarios": sanitized_scenarios,
        },
        trace_id=_current_trace_id(),
    )


@router.get("/audit", summary="Tenant-scoped SafeHarness audit")
async def get_safeharness_audit(
    request: Request,
    tenant_id: Optional[str] = Query(
        None, min_length=1, max_length=256, pattern=_IDENTIFIER_PATTERN
    ),
    source: Optional[Literal["agent", "gateway"]] = None,
    action: Optional[str] = Query(None, min_length=1, max_length=256, pattern=_IDENTIFIER_PATTERN),
    limit: int = Query(50, ge=1, le=200),
    principal: SafeHarnessPrincipal = Depends(_require_view),
):
    _validate_audit_query(request)
    if tenant_id is not None:
        _require_same_tenant(tenant_id, principal)
    params = {"tenant_id": principal.tenant_id, "limit": str(limit)}
    if source is not None:
        params["source"] = source
    if action is not None:
        params["action"] = action
    try:
        return success_response(
            data=await _upstream_request("GET", "audit", params=params),
            trace_id=_current_trace_id(),
        )
    except SafeHarnessUpstreamError as exc:
        raise _upstream_unavailable() from exc


@router.get("/policies", summary="Read-only SafeHarness policy packages")
async def get_safeharness_policies(
    principal: SafeHarnessPrincipal = Depends(_require_view),
):
    return success_response(
        data={
            "schema_version": "1.0",
            "tenant_id": principal.tenant_id,
            "packages": list(_POLICY_PACKAGES),
            "diff": {
                "status": "unavailable",
                "reason": "No candidate policy package is configured.",
            },
            "publishing": {"available": False},
            "rollback": {"available": False},
        },
        trace_id=_current_trace_id(),
    )


@router.post(
    "/policies/simulate",
    summary="Simulate a current SafeHarness policy package",
    openapi_extra=_POLICY_SIMULATION_OPENAPI,
)
async def simulate_safeharness_policy(
    request: Request,
    principal: SafeHarnessPrincipal = Depends(_require_view),
):
    simulation = await _read_json_model(request, PolicySimulationRequest)
    package = next(item for item in _POLICY_PACKAGES if item["id"] == simulation.package_id)
    if simulation.operation not in package["operations"]:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "safeharness_policy_operation_mismatch",
                "message": "The operation is not governed by the selected policy package.",
            },
        )
    approval_required = simulation.operation in {
        operation for package in _POLICY_PACKAGES for operation in package["operations"]
    }
    return success_response(
        data={
            "schema_version": "1.0",
            "tenant_id": principal.tenant_id,
            "package_id": package["id"],
            "operation": simulation.operation,
            "mode": "deterministic_read_only",
            "decision": "require_approval" if approval_required else "deny",
            "impact": {
                "requires_reauth": simulation.operation != "internet_search",
                "requires_one_shot_approval": approval_required,
                "writes_control_plane": simulation.operation != "internet_search",
                "gateway_verified": False,
                "production_ready": False,
            },
        },
        trace_id=_current_trace_id(),
    )


@router.post(
    "/mutations",
    summary="Prepare a controlled SafeHarness mutation",
    openapi_extra=_MUTATION_PREPARE_OPENAPI,
)
async def prepare_safeharness_mutation(
    request: Request,
    principal: SafeHarnessPrincipal = Depends(_require_manage),
    db: Session = Depends(get_db),
):
    operation = await _read_json_model(request, MutationPrepareRequest)
    try:
        item, created = mutation_coordinator.prepare(
            operation,
            operator_id=principal.user_id,
            authoritative_tenant_id=principal.tenant_id,
            db=db,
        )
    except PermissionError as exc:
        code = (
            "safeharness_cross_tenant"
            if operation.target.tenant_id != principal.tenant_id
            else "safeharness_reauth_failed"
        )
        raise HTTPException(
            status_code=403,
            detail={"code": code, "message": "Mutation authorization failed."},
        ) from exc
    except MutationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "safeharness_idempotency_conflict",
                "message": "The idempotency key is already bound to another mutation.",
            },
        ) from exc
    return success_response(
        data={
            **_decision_payload(item, status=item.state),
            "created": created,
            "approval": {
                "status": item.state,
                "one_shot": True,
                "expires_in_seconds": APPROVAL_TTL_SECONDS,
            },
        },
        trace_id=_current_trace_id(),
    )


@router.post(
    "/mutations/{approval_id}/decision",
    summary="Decide and execute a controlled SafeHarness mutation",
    openapi_extra=_MUTATION_DECISION_OPENAPI,
)
async def decide_safeharness_mutation(
    request: Request,
    approval_id: str = Path(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
    principal: SafeHarnessPrincipal = Depends(_require_manage),
    db: Session = Depends(get_db),
):
    decision = await _read_json_model(request, MutationDecisionRequest)
    try:
        item, execute = mutation_coordinator.begin_decision(
            approval_id,
            decision,
            operator_id=principal.user_id,
            authoritative_tenant_id=principal.tenant_id,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "safeharness_approval_mismatch",
                "message": "The approval is not bound to this operator and target.",
            },
        ) from exc
    except MutationUnavailable as exc:
        raise HTTPException(
            status_code=410,
            detail={
                "code": "safeharness_approval_unavailable",
                "message": "The one-shot approval is missing or expired.",
            },
        ) from exc
    except MutationIndeterminate as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "safeharness_mutation_indeterminate",
                "message": "The mutation outcome is unknown and requires reconciliation.",
            },
        ) from exc
    except MutationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "safeharness_mutation_conflict",
                "message": "The mutation is already being processed.",
            },
        ) from exc

    if not execute:
        return success_response(data=item.result, trace_id=_current_trace_id())

    if item.action.startswith(("capability.", "plugin.", "automation.")):
        try:
            result = await _execute_sumc_mutation(item, db)
            mutation_coordinator.complete(item, result)
            return success_response(data=result, trace_id=_current_trace_id())
        except SafeHarnessUpstreamError as exc:
            result = {
                **_decision_payload(item, status="rejected"),
                "approval": {"status": "approved", "one_shot": True},
                "reason_code": "safeharness_control_plane_unavailable",
                "receipt": None,
            }
            mutation_coordinator.complete(item, result)
            return success_response(data=result, trace_id=_current_trace_id())
        except (BadRequestError, ResourceNotFoundError, HTTPException) as exc:
            status_code = exc.status_code
            result = {
                **_decision_payload(item, status="rejected"),
                "approval": {"status": "approved", "one_shot": True},
                "reason_code": (
                    "safeharness_execution_unavailable"
                    if status_code == 503
                    else "safeharness_target_unavailable"
                ),
                "receipt": None,
            }
            mutation_coordinator.complete(item, result)
            return success_response(data=result, trace_id=_current_trace_id())
        except Exception as exc:
            mutation_coordinator.indeterminate(item)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "safeharness_mutation_indeterminate",
                    "message": "The mutation outcome is unknown and requires reconciliation.",
                },
            ) from exc

    route, expected_operation = {
        "agent.quarantine": ("quarantine", "quarantine_agent"),
        "agent.restore": ("restore", "restore_agent"),
        "delegation.revoke": ("delegation_revoke", "revoke_delegation"),
    }[item.action]
    try:
        upstream = await _upstream_request(
            "POST",
            route,
            payload={
                "schema_version": "1.0",
                "action": item.action,
                "tenant_id": item.tenant_id,
                **item.target,
                "resource": item.resource,
                "effect": item.effect,
                "operator_id": item.operator_id,
                "reason": item.reason,
                "decision_id": item.decision_id,
                "approval_id": item.approval_id,
                "policy_version": "safeharness.mutations.v1",
                "idempotency_key": item.idempotency_key,
            },
        )
        signature = upstream.get("signature")
        version = upstream.get("version")
        if (
            not isinstance(signature, str)
            or re.fullmatch(r"[0-9a-f]{64}", signature) is None
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
            or upstream.get("operation") != expected_operation
        ):
            raise SafeHarnessUpstreamError("SafeHarness mutation receipt is invalid")
        result = {
            **_decision_payload(item, status="completed"),
            "approval": {"status": "approved", "one_shot": True},
            "receipt": {
                "receipt_id": signature,
                "source": "coordination_mutation_log",
                "version": version,
                "integrity": "signed_mutation_log",
                "gateway_verified": False,
                "production_ready": False,
            },
        }
        mutation_coordinator.complete(item, result)
        return success_response(data=result, trace_id=_current_trace_id())
    except SafeHarnessUpstreamError as exc:
        mutation_coordinator.indeterminate(item)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "safeharness_mutation_indeterminate",
                "message": "The mutation outcome is unknown and requires reconciliation.",
            },
        ) from exc


def _application_authorization_envelope(item) -> dict[str, str]:
    return {
        "schema_version": "1.0",
        "action": item.action,
        "tenant_id": item.tenant_id,
        **item.target_values,
        "resource": item.resource,
        "effect": item.effect,
        "operator_id": item.operator_id,
        "reason": item.reason,
        "decision_id": item.decision_id,
        "approval_id": item.approval_id,
        "policy_version": "safeharness.mutations.v1",
        "idempotency_key": item.idempotency_key,
    }


async def _authorize_application_mutation(item) -> dict[str, str]:
    envelope = _application_authorization_envelope(item)
    authorization = await _upstream_request(
        "POST", "application_authorize", payload=envelope
    )
    encoded = json.dumps(
        envelope, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected_digest = hashlib.sha256(encoded).hexdigest()
    expected_signature = hmac.new(
        settings.safeharness.mutation_key.encode("utf-8"),
        b"safeharness-application-authorization-v1\0" + encoded,
        hashlib.sha256,
    ).hexdigest()
    signature = authorization.get("signature")
    audit_reference = f"application-authorization:{signature}"
    if not (
        authorization.get("schema_version") == "1.0"
        and authorization.get("operation") == "authorize_application_mutation"
        and authorization.get("action") == item.action
        and authorization.get("envelope_digest") == expected_digest
        and authorization.get("authorization_id") == signature
        and authorization.get("audit_reference") == audit_reference
        and authorization.get("integrity") == "hmac_envelope_signature"
        and authorization.get("persistent") is False
        and isinstance(signature, str)
        and hmac.compare_digest(signature, expected_signature)
        and authorization.get("gateway_verified") is False
        and authorization.get("production_ready") is False
    ):
        raise SafeHarnessUpstreamError("SafeHarness application authorization is invalid")
    return authorization


async def _execute_sumc_mutation(item, db: Session) -> dict:
    """Execute only server-selected application services after one-shot approval."""
    authorization = await _authorize_application_mutation(item)
    status = "completed"
    if item.action.startswith("capability."):
        from api.routes.v1.catalog import UpdateCatalogRequest, _update_catalog_item_for_user

        response = await _update_catalog_item_for_user(
            kind=item.target_values["kind"],
            id=item.target_values["item_id"],
            request=UpdateCatalogRequest(enabled=item.action.endswith(".enable")),
            user_id=item.operator_id,
            db=db,
        )
        application_result = response["data"]
    elif item.action.startswith("plugin."):
        from core.services.plugin_service import set_plugin_enabled_for_user

        application_result = set_plugin_enabled_for_user(
            db,
            item.target_values["install_id"],
            enabled=item.action.endswith(".enable"),
            user_id=item.operator_id,
        )
    else:
        from core.services.automation_service import AutomationService
        from orchestration.schedulers.automation_scheduler import get_scheduler

        task = AutomationService(db).get_task(item.target_values["task_id"], item.operator_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Automation not found")
        if task.status not in ("active", "paused"):
            raise HTTPException(status_code=409, detail="Automation cannot be triggered")
        scheduler = get_scheduler()
        if scheduler is None:
            raise HTTPException(status_code=503, detail="Automation scheduler unavailable")
        asyncio.create_task(scheduler.execute_task(task.task_id, task.user_id))
        status = "accepted"
        application_result = {"task_id": task.task_id, "execution_status": "scheduled"}

    return {
        **_decision_payload(item, status=status),
        "approval": {"status": "approved", "one_shot": True},
        "application_result": application_result,
        "receipt": {
            "receipt_id": authorization["authorization_id"],
            "source": "safeharness_application_authorization",
            "authorization_id": authorization["authorization_id"],
            "audit_reference": authorization["audit_reference"],
            "envelope_digest": authorization["envelope_digest"],
            "integrity": "hmac_envelope_signature",
            "persistent": False,
            "gateway_verified": False,
            "production_ready": False,
        },
    }


__all__ = [
    "SafeHarnessChatRequest",
    "SafeHarnessPrincipal",
    "capability_discovery_payload",
    "protected_chat_available",
    "register_protected_chat_stream",
    "resolve_safeharness_principal",
    "router",
    "safeharness_error_response",
    "unregister_protected_chat_stream",
    "validate_discovery_query",
]
