"""Agent-side bridge for gateway-routed tools of external framework agents.

The Tool Gateway owns the executors and the capability signing key; an external
framework agent only gets a thin callable per tool. Executing one tool call
walks the full proof chain the gateway enforces:

1. build the exact ``ToolInvocation``;
2. sign a control-plane enforcement receipt with the agent-side issuer built
   from the same environment the runtime uses (fail-closed: without receipt
   signing material the tool surface is not mounted at all, because the gateway
   requires a receipt on every execution);
3. obtain a one-invocation capability grant from the gateway;
4. execute through the gateway (atomic consume of both proofs server-side).

No bundle, no tools: when ``SAFEHARNESS_TOOL_GATEWAY_URL`` is unset the
external agents keep running as pure reasoners. An unreachable gateway or
missing signing material also degrades to "no tools" — never to local
execution, which would bypass the authorization split.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

POLICY_PACKAGE_ID = "safeharness.default.v1"


def _descriptor():
    from core.safeharness_tool_gateway import _demo_tool

    return _demo_tool().descriptor


async def build_gateway_tool_bundle(
    *,
    session_id: str,
    run_id: str,
    environ: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Probe the deployment and return a mutable execution bundle, or None.

    None means "tool surface not mounted" — gateway unconfigured, unreachable,
    or receipt signing material missing. Never raises for an absent gateway:
    external agents must keep working as pure reasoners.
    """
    from safeharness.api.context import SessionContext
    from safeharness.backends.receipt_config import (
        build_enforcement_receipt_issuer_from_environment,
    )

    values = os.environ if environ is None else environ
    from core.safeharness_tool_gateway import (
        GATEWAY_DEMO_TOOL_ID,
        gateway_client_from_environment,
    )

    client = gateway_client_from_environment(values)
    if client is None:
        return None
    issuer = build_enforcement_receipt_issuer_from_environment(values)
    if issuer is None:
        logger.warning(
            "[external-agents] gateway URL set without receipt signing material; "
            "tool surface stays unmounted (execution would fail closed)"
        )
        return None
    try:
        await client.health()
    except Exception as exc:  # noqa: BLE001 — unreachable gateway is a soft skip
        logger.warning(
            "[external-agents] tool gateway health probe failed (%s); "
            "tool surface stays unmounted",
            type(exc).__name__,
        )
        return None
    # The gateway authorises the *platform* execution identity — the exact
    # tenant/agent pair its capability mapping was built with — and rejects any
    # other subject ("cannot act as this tenant/Agent"). The subject therefore
    # comes from the gateway's own environment defaults, never from request
    # context (a user-facing tenant string would not match the principal).
    # The originating framework agent is carried in metadata and evidence only.
    from core.safeharness_tool_gateway import (
        GATEWAY_DEFAULT_ISSUER_AGENT,
        GATEWAY_DEFAULT_ISSUER_TENANT,
    )

    principal_tenant = values.get("SAFEHARNESS_DEMO_TENANT_ID", "").strip() or GATEWAY_DEFAULT_ISSUER_TENANT
    principal_agent = values.get("SAFEHARNESS_COMMUNICATION_AGENT_ID", "").strip() or GATEWAY_DEFAULT_ISSUER_AGENT
    return {
        "tool_name": GATEWAY_DEMO_TOOL_ID,
        "client": client,
        "issuer": issuer,
        "context": SessionContext(
            tenant_id=principal_tenant,
            agent_id=principal_agent,
            session_id=session_id,
            run_id=run_id,
            framework="sumc-external-agents",
            trace_id=str(uuid4()),
        ),
        "descriptor": _descriptor(),
        "calls": [],
    }


async def execute_gateway_tool(bundle: dict[str, Any]) -> dict[str, Any]:
    """One full-proof execution of the demo tool through the gateway.

    Gateway rejections come back as ``{"error": {code, message}}`` instead of
    raising: the framework loop feeds the error text to the model, which is the
    same contract a real tool has — one tool failing must not kill the run.
    """
    from safeharness.api.tools import ToolInvocation
    from safeharness.gateway.client import ToolGatewayError

    client = bundle["client"]
    issuer = bundle["issuer"]
    context = bundle["context"]
    descriptor = bundle["descriptor"]
    invocation = ToolInvocation(
        tool_name=descriptor.name,
        arguments={},
        tool_version=descriptor.version,
    )
    receipt = await issuer.issue(
        context,
        descriptor,
        invocation,
        decision_id=f"ext-tool-{uuid4().hex[:12]}",
        event_id=f"ext-tool-event-{uuid4().hex[:12]}",
        policy_version=POLICY_PACKAGE_ID,
        workload_id="",
    )
    grant = await client.issue(context, descriptor, invocation)
    bound = ToolInvocation(
        tool_name=invocation.tool_name,
        arguments=invocation.arguments,
        call_id=invocation.call_id,
        tool_version=invocation.tool_version,
        capability_token=grant.token,
        enforcement_receipt=receipt,
    )
    started = time.monotonic()
    try:
        result = await client.execute(context, bound)
    except ToolGatewayError as exc:
        bundle["calls"].append(
            {
                "tool": descriptor.name,
                "call_id": bound.call_id,
                "ok": False,
                "error_code": exc.code,
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
        )
        return {"error": {"code": exc.code, "message": str(exc)}}
    bundle["calls"].append(
        {
            "tool": descriptor.name,
            "call_id": bound.call_id,
            "ok": True,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "receipt": "verified",
            "gateway": (os.environ.get("SAFEHARNESS_TOOL_GATEWAY_URL") or "").rstrip("/"),
        }
    )
    return result if isinstance(result, dict) else {"result": result}


def execute_gateway_tool_in_worker(bundle: dict[str, Any]) -> dict[str, Any]:
    """Blocking form for sync framework loops (smolagents worker thread).

    The thread has no running event loop, so ``asyncio.run`` is safe there.
    """
    return asyncio.run(execute_gateway_tool(bundle))


def tool_snapshot(bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Evidence view of the bundle's tool calls (empty when unmounted)."""
    if not bundle:
        return []
    return list(bundle.get("calls") or [])


__all__ = [
    "build_gateway_tool_bundle",
    "execute_gateway_tool",
    "execute_gateway_tool_in_worker",
    "tool_snapshot",
]
