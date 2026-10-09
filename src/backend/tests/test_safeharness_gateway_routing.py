"""Protected-chat gateway routing — env-driven split of issuance and execution.

When ``SAFEHARNESS_TOOL_GATEWAY_URL`` is configured, protected-chat tool calls
issue capabilities from the gateway and execute through it (the local runtime
still decides and countersigns; the gateway verifies both proofs). Without the
URL the local path is untouched. These tests pin the routing pair contract and
walk the full proof chain over a real HTTP gateway with the read-only demo
tool (the same registry the demo gateway process serves).
"""

import os
from threading import Thread

import pytest

from safeharness import SessionContext
from safeharness.gateway.service import create_gateway_server

from core.safeharness_protected_chat import build_gateway_routing
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


def test_routing_is_none_without_gateway(monkeypatch):
    monkeypatch.delenv("SAFEHARNESS_TOOL_GATEWAY_URL", raising=False)
    assert build_gateway_routing(None) is None


async def test_routing_pair_binds_executor_to_issued_context():
    """The executor must reuse the exact context the issuer saw."""
    recorded: dict = {}

    class _FakeClient:
        async def issue(self, context, descriptor, invocation):
            recorded["issue_context"] = context
            from safeharness.ports import CapabilityGrant
            from datetime import datetime, timezone

            return CapabilityGrant(
                token="fake-token", token_id="t1", expires_at=datetime.now(timezone.utc),
                subject=context.agent_id, tenant_id=context.tenant_id,
                session_id=context.session_id, run_id=context.run_id,
                tool_name=descriptor.name, tool_version=descriptor.version,
                tool_integrity_digest=descriptor.integrity_digest,
                invocation_digest="digest",
            )

        async def execute(self, context, invocation):
            recorded["execute_context"] = context
            return {"ok": True}

    routing = build_gateway_routing(_FakeClient())
    assert routing is not None
    issuer, executor = routing
    context = SessionContext(tenant_id="fiberhome-demo", agent_id="main_agent")
    from safeharness import ToolDescriptor, ToolInvocation
    from safeharness.api.tools import Reversibility

    descriptor = ToolDescriptor(
        name=GATEWAY_DEMO_TOOL_ID, version="1", description="d",
        input_schema={"type": "object", "properties": {}},
        required_capabilities=frozenset({"platform:status:read"}),
        side_effects=frozenset(), data_classifications=frozenset(),
        reversibility=Reversibility.READ_ONLY, idempotent=True,
        sandbox_required=False,
    )
    await issuer.issue(context, descriptor, ToolInvocation(tool_name=descriptor.name))
    with pytest.raises(RuntimeError, match="before capability issuance"):
        routing_fresh = build_gateway_routing(_FakeClient())
        assert routing_fresh is not None
        await routing_fresh[1](ToolInvocation(tool_name=descriptor.name))
    result = await executor(ToolInvocation(tool_name=descriptor.name))
    assert result == {"ok": True}
    assert recorded["execute_context"] is recorded["issue_context"]


async def test_routing_full_chain_over_real_http(tmp_path, monkeypatch):
    """Issue through the routing pair, execute through the real gateway."""
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
        monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_URL", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv("SAFEHARNESS_TOOL_GATEWAY_API_KEY", _API_KEY)
        from safeharness.gateway.client import RemoteToolGatewayClient

        from core.safeharness_capability_mapping import build_internet_search_tool

        client = RemoteToolGatewayClient(
            os.environ["SAFEHARNESS_TOOL_GATEWAY_URL"], _API_KEY
        )
        routing = build_gateway_routing(client)
        assert routing is not None
        issuer, executor = routing
        # The descriptor must match what the gateway registered — the same
        # builder constructs both sides.
        descriptor = build_internet_search_tool().descriptor
        context = SessionContext(
            tenant_id="fiberhome-demo", agent_id="main_agent",
            framework="sumc-protected-chat",
        )
        from safeharness import ToolInvocation

        grant = await issuer.issue(
            context,
            descriptor,
            ToolInvocation(
                tool_name=descriptor.name,
                arguments={"query": "alarm"},
                tool_version=descriptor.version,
            ),
        )
        assert grant.token
        # Execution through the bare routing pair carries no enforcement receipt
        # (in the real protected-chat flow the local runtime's around_tool signs
        # one after the policy decision) — the gateway must refuse it:
        # fail-closed, exactly as designed.
        from safeharness.gateway.client import ToolGatewayError

        with pytest.raises(ToolGatewayError) as excinfo:
            await executor(
                ToolInvocation(
                    tool_name=descriptor.name,
                    arguments={"query": "alarm"},
                    tool_version=descriptor.version,
                )
            )  # version pinned: the gateway requires an exact match
        assert excinfo.value.code == "enforcement_receipt_missing"
    finally:
        server.server_close()
        thread.join(timeout=5)
