"""External-agent tool surface routed through the central Tool Gateway.

The demo tool ``system_status_snapshot`` is registered server-side only; the
external-agent bridge mounts it on the three framework runners only when the
gateway deployment is fully configured. These tests pin: the server registry
carries the read-only demo tool, a bridge execution walks the whole proof
chain (receipt + capability + gateway execute) over real HTTP, a replayed
proof is rejected, and an incomplete deployment keeps the surface unmounted.
"""

from threading import Thread

import pytest

from safeharness import SessionContext
from safeharness.gateway.client import RemoteToolGatewayClient, ToolGatewayError
from safeharness.gateway.protocol import request_to_wire
from safeharness.gateway.service import create_gateway_server

from core.safeharness_tool_gateway import (
    GATEWAY_DEMO_TOOL_ID,
    build_gateway_application,
)

_API_KEY = "gateway-api-key-0123456789abcdef"
_CAPABILITY_KEY = "gateway-capability-key-0123456789abcdef"


def _gateway_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_ENFORCEMENT_RECEIPT_MODE", "hmac")
    monkeypatch.setenv("SAFEHARNESS_ENFORCEMENT_RECEIPT_SIGNING_KEY", "receipt-hmac-key-0123456789abcdef")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_CAPABILITY_KEY", _CAPABILITY_KEY)
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_REPLAY_DB", str(tmp_path / "replay.db"))
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_AUDIT_DB", str(tmp_path / "gateway-audit.db"))
    monkeypatch.setenv("SAFEHARNESS_DEMO_TENANT_ID", "fiberhome-demo")
    monkeypatch.setenv("SAFEHARNESS_COMMUNICATION_AGENT_ID", "main_agent")


async def test_gateway_registry_carries_demo_tool_and_executes_it(tmp_path, monkeypatch):
    _gateway_env(tmp_path, monkeypatch)
    application = build_gateway_application()
    from safeharness.api.tools import ToolInvocation

    tool = application.registry.resolve(ToolInvocation(tool_name=GATEWAY_DEMO_TOOL_ID))
    assert tool is not None
    context = SessionContext(
        tenant_id="fiberhome-demo",
        agent_id="main_agent",
        session_id="session-d",
        run_id="run-d",
        framework="sumc-gateway-demo-test",
    )
    from safeharness.api.tools import ToolInvocation

    from safeharness.backends.receipt_config import (
        build_enforcement_receipt_issuer_from_environment,
    )

    invocation = ToolInvocation(tool_name=tool.descriptor.name, tool_version=tool.descriptor.version)
    issuer = build_enforcement_receipt_issuer_from_environment()
    assert issuer is not None
    receipt = await issuer.issue(
        context, tool.descriptor, invocation,
        decision_id="decision-demo", event_id="event-demo",
        policy_version="safeharness.default.v1", workload_id="",
    )
    grant = await application.capability_gateway.issue(context, tool.descriptor, invocation)
    bound = ToolInvocation(
        tool_name=invocation.tool_name,
        call_id=invocation.call_id,
        tool_version=invocation.tool_version,
        capability_token=grant.token,
        enforcement_receipt=receipt,
    )
    status, response = await application.execute(request_to_wire(context, bound))
    assert status == 200, (status, response)
    result = response["result"]
    assert result["python_version"] and result["platform"]
    assert "T" in result["utc_now"]

    # The verification side must leave its own trace: the gateway audit ledger
    # records the executed (and replay-protected) tool call.
    import sqlite3

    rows = sqlite3.connect(str(tmp_path / "gateway-audit.db")).execute(
        "SELECT action, outcome, tool_name FROM safeharness_gateway_audit "
        "ORDER BY id"
    ).fetchall()
    assert ("tool_execute", "succeeded", GATEWAY_DEMO_TOOL_ID) in rows, rows


async def test_bridge_full_proof_round_trip_over_real_http(tmp_path, monkeypatch):
    from core.external_agents.gateway_tools import (
        build_gateway_tool_bundle,
        execute_gateway_tool,
        tool_snapshot,
    )

    _gateway_env(tmp_path, monkeypatch)
    server = create_gateway_server(
        build_gateway_application(),
        host="127.0.0.1",
        port=0,
        api_key=_API_KEY,
        tenant_id="fiberhome-demo",
        agent_id="main_agent",
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_URL", url)
        monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_API_KEY", _API_KEY)
        bundle = await build_gateway_tool_bundle(
            session_id="sess-ext",
            run_id="run-ext",
        )
        assert bundle is not None
        result = await execute_gateway_tool(bundle)
        assert result["platform"] in {"Windows", "Linux", "Darwin"}
        calls = tool_snapshot(bundle)
        assert len(calls) == 1
        assert calls[0]["tool"] == GATEWAY_DEMO_TOOL_ID
        assert calls[0]["receipt"] == "verified"
    finally:
        server.server_close()
        thread.join(timeout=5)


async def test_bridge_replayed_proof_is_rejected(tmp_path, monkeypatch):
    """A second execution reusing call one's exact proofs is rejected."""
    from safeharness.api.tools import ToolInvocation
    from safeharness.backends.receipt_config import (
        build_enforcement_receipt_issuer_from_environment,
    )

    _gateway_env(tmp_path, monkeypatch)
    server = create_gateway_server(
        build_gateway_application(),
        host="127.0.0.1",
        port=0,
        api_key=_API_KEY,
        tenant_id="fiberhome-demo",
        agent_id="main_agent",
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_URL", url)
        monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_API_KEY", _API_KEY)
        from core.external_agents.gateway_tools import (
            build_gateway_tool_bundle,
            execute_gateway_tool,
        )

        bundle = await build_gateway_tool_bundle(
            session_id="sess-replay",
            run_id="run-replay",
        )
        assert bundle is not None
        first = await execute_gateway_tool(bundle)
        assert first

        # Rebuild the same proof chain manually, execute once, then replay.
        client = bundle["client"]
        context = bundle["context"]
        descriptor = bundle["descriptor"]
        issuer = bundle["issuer"]
        invocation = ToolInvocation(tool_name=descriptor.name, tool_version=descriptor.version)
        receipt = await issuer.issue(
            context, descriptor, invocation,
            decision_id="decision-replay", event_id="event-replay",
            policy_version="safeharness.default.v1", workload_id="",
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
        assert await client.execute(context, bound) is not None
        with pytest.raises(ToolGatewayError) as excinfo:
            await client.execute(context, bound)
        assert excinfo.value.code in {
            "capability_token_replayed",
            "enforcement_receipt_replayed",
        }
    finally:
        server.server_close()
        thread.join(timeout=5)


async def test_bridge_stays_unmounted_without_gateway_or_key(tmp_path, monkeypatch):
    from core.external_agents.gateway_tools import build_gateway_tool_bundle

    monkeypatch.delenv("SAFEHARNESS_TOOL_GATEWAY_URL", raising=False)
    assert await build_gateway_tool_bundle(session_id="s", run_id="r") is None

    _gateway_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_API_KEY", _API_KEY)
    monkeypatch.delenv("SAFEHARNESS_ENFORCEMENT_RECEIPT_MODE", raising=False)
    monkeypatch.delenv("SAFEHARNESS_ENFORCEMENT_RECEIPT_SIGNING_KEY", raising=False)
    assert await build_gateway_tool_bundle(session_id="s", run_id="r") is None


async def test_three_runner_tool_adapters_construct(tmp_path, monkeypatch):
    """The native tool wrappers build against the installed frameworks."""
    _gateway_env(tmp_path, monkeypatch)
    from core.external_agents.runners import (
        _autogen_tool,
        _openai_agents_tool,
        _smolagents_tool,
    )

    bundle = {"calls": []}
    s_tool = _smolagents_tool(bundle)
    assert s_tool.name == GATEWAY_DEMO_TOOL_ID
    assert s_tool.inputs == {}

    o_tool = await _openai_agents_tool(bundle)
    assert o_tool.name == GATEWAY_DEMO_TOOL_ID
    assert o_tool.params_json_schema["type"] == "object"

    a_tool = await _autogen_tool(bundle)
    assert a_tool.name == GATEWAY_DEMO_TOOL_ID
