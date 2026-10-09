"""Contract checks for the fail-closed capability read model."""

import asyncio

from core.safeharness_capabilities import with_capability_security
from core.safeharness_capability_mapping import build_internet_search_tool, build_local_runtime
from safeharness.api.context import SessionContext
from safeharness.api.tools import ToolInvocation


def test_capability_read_model_keeps_unmapped_security_facts_unknown():
    item = with_capability_security(
        {"id": "search", "name": "Search", "security": {"execution": "allowed"}},
        kind="mcp",
        identifier_key="id",
        source_type="runtime_catalog",
    )

    assert item["security"]["source"] == {
        "type": "runtime_catalog",
        "kind": "mcp",
        "id": "search",
    }
    assert item["security"]["tool_descriptor"] == {"status": "unknown"}
    assert item["security"]["adapter_coverage"] == {"status": "unknown"}
    assert item["security"]["policy_package"] == {"status": "unknown"}
    assert item["security"]["execution"]["status"] == "blocked"
    assert "required_capabilities" not in item["security"]
    assert "side_effects" not in item["security"]


def test_capability_read_model_maps_only_verified_internet_search():
    item = with_capability_security(
        {"id": "internet_search", "name": "Internet search"},
        kind="mcp",
        identifier_key="id",
        source_type="runtime_catalog",
    )

    descriptor = item["security"]["tool_descriptor"]
    assert descriptor["status"] == "mapped"
    assert descriptor["name"] == "internet_search"
    assert descriptor["version"] == "1"
    assert descriptor["required_capabilities"] == ["internet:search"]
    assert descriptor["side_effects"] == ["network_egress"]
    assert descriptor["data_classifications"] == ["public_web", "user_query"]
    assert descriptor["reversibility"] == "read_only"
    assert descriptor["idempotent"] is True
    assert descriptor["compensator"] is None
    assert descriptor["sandbox_required"] is False
    assert len(descriptor["integrity_digest"]) == 64
    assert item["security"]["adapter_coverage"]["status"] == "mapped"
    assert item["security"]["policy_package"]["id"] == "safeharness.default.v1"
    assert item["security"]["execution"] == {
        "status": "approval_required",
        "reason_code": "safeharness_one_shot_approval_required",
        "production_ready": False,
    }


def test_internet_search_mapping_builds_read_only_descriptor():
    tool = build_internet_search_tool()
    descriptor = tool.descriptor

    assert descriptor.name == "internet_search"
    assert descriptor.version == "1"
    assert descriptor.required_capabilities == frozenset({"internet:search"})
    assert descriptor.side_effects == frozenset({"network_egress"})
    assert descriptor.reversibility.value == "read_only"
    assert descriptor.idempotent is True
    assert descriptor.sandbox_required is False
    assert descriptor.input_schema["type"] == "object"
    assert tuple(descriptor.input_schema["required"]) == ("query",)
    assert set(descriptor.input_schema["properties"]) == {"query", "max_results"}

    runtime = build_local_runtime(tool, agent_id="agent-1")
    assert runtime.services.tool_catalog is not None
    assert runtime.services.capability_authority is not None
    assert runtime.policy_engine.policy_version == "safeharness.default.v1"


def test_internet_search_mapping_authorizes_exact_agent_and_denies_other_agent():
    tool = build_internet_search_tool()
    runtime = build_local_runtime(tool, agent_id="agent-1")
    authority = runtime.services.capability_authority
    assert authority is not None
    invocation = ToolInvocation(
        tool_name="internet_search",
        tool_version="1",
        arguments={"query": "SafeHarness", "max_results": 1},
    )

    async def authorize():
        allowed_context = SessionContext(tenant_id="tenant-1", agent_id="agent-1")
        grant = await authority.issue(allowed_context, tool.descriptor, invocation)
        signed_invocation = ToolInvocation(
            tool_name=invocation.tool_name,
            tool_version=invocation.tool_version,
            arguments=invocation.arguments,
            call_id=invocation.call_id,
            capability_token=grant.token,
        )
        return (
            await authority.authorize(
                allowed_context,
                tool.descriptor,
                signed_invocation,
            ),
            await authority.authorize(
                SessionContext(tenant_id="tenant-1", agent_id="agent-2"),
                tool.descriptor,
                signed_invocation,
            ),
        )

    allowed, denied = asyncio.run(authorize())

    assert allowed.allowed is True
    assert denied.allowed is False


def test_internet_search_mapping_fails_closed_without_approval_backend(monkeypatch):
    from langchain_core.messages import AIMessage
    from safeharness.adapters.langgraph import LangGraphAgent
    from safeharness.chat_bridge import ProtectedChatRequest, ProtectedChatRunner
    from mcp_servers.internet_search_mcp import server

    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return {"result": [{"title": "SafeHarness", "url": "https://example.invalid"}]}

    class ToolModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, schemas):
            assert schemas[0]["function"]["name"] == "internet_search"
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "search-1",
                            "name": "internet_search",
                            "args": {"query": "SafeHarness", "max_results": 1},
                        }
                    ],
                )
            return AIMessage(content="Search completed safely.")

    monkeypatch.setattr(server, "internet_search", fake_search)
    tool = build_internet_search_tool()
    runtime = build_local_runtime(tool, agent_id="agent-1")
    authority = runtime.services.capability_authority
    runner = ProtectedChatRunner(
        lambda _request: LangGraphAgent(
            model=ToolModel(),
            tools=(tool,),
            capability_issuer=authority,
        ),
        runtime=runtime,
        security_timeline=True,
    )

    result = asyncio.run(
        runner.ainvoke(
            ProtectedChatRequest(
                tenant_id="tenant-1",
                user_id="user-1",
                role="member",
                conversation_id="chat-1",
                agent_id="agent-1",
                message="Search for SafeHarness.",
            )
        )
    )

    assert calls == [], result.events
    event_types = [event["type"] for event in result.events]
    assert "tool_proposal" in event_types
    assert "approval" in event_types
    assert "execution" not in event_types
    proposed = [
        event
        for event in result.events
        if event["type"] == "policy_decision"
        and event["data"]["event_kind"] == "tool_call_proposed"
    ]
    assert [event["data"]["disposition"] for event in proposed] == [
        "require_approval",
        "deny",
    ]
    assert any(
        event["type"] == "approval" and event["data"]["status"] == "denied"
        for event in result.events
    )


def test_safeharness_approval_is_one_shot():
    import core.safeharness_approval as confirmation

    async def run():
        task = asyncio.create_task(
            confirmation.approval_gate(
                chat_id="chat-approval",
                tool_name="internet_search",
                decision_id="decision-1",
                timeout=1,
            )
        )
        await asyncio.sleep(0)
        pending = confirmation.get_pending("chat-approval")
        assert pending is not None
        rejected = confirmation.set_decision(
            "chat-approval",
            pending["confirm_id"],
            "allow_session",
            approver="user-1",
        )
        assert rejected["ok"] is False
        approved = confirmation.set_decision(
            "chat-approval",
            pending["confirm_id"],
            confirmation.ALLOW,
            approver="user-1",
        )
        assert approved["ok"] is True
        return await task

    result = asyncio.run(run())
    assert result["approved"] is True
    assert result["approver"] == "user-1"
    assert result["approval_id"]


def test_internet_search_executes_once_after_authenticated_approval(monkeypatch):
    from langchain_core.messages import AIMessage
    import core.safeharness_approval as confirmation
    from core.safeharness_approval import SUMCApprovalBroker
    from mcp_servers.internet_search_mcp import server
    from safeharness.adapters.langgraph import LangGraphAgent
    from safeharness.chat_bridge import ProtectedChatRequest, ProtectedChatRunner

    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return {"result": [{"title": "SafeHarness", "url": "https://example.invalid"}]}

    class ToolModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, _schemas):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "id": "search-approved",
                        "name": "internet_search",
                        "args": {"query": "SafeHarness", "max_results": 1},
                    }],
                )
            return AIMessage(content="Search completed safely.")

    monkeypatch.setattr(server, "internet_search", fake_search)
    tool = build_internet_search_tool()
    runtime = build_local_runtime(
        tool,
        agent_id="agent-1",
        approval_broker=SUMCApprovalBroker(),
    )
    runner = ProtectedChatRunner(
        lambda _request: LangGraphAgent(
            model=ToolModel(),
            tools=(tool,),
            capability_issuer=runtime.services.capability_authority,
        ),
        runtime=runtime,
        security_timeline=True,
    )

    async def run():
        task = asyncio.create_task(
            runner.ainvoke(
                ProtectedChatRequest(
                    tenant_id="tenant-1",
                    user_id="user-1",
                    role="member",
                    conversation_id="chat-approved",
                    agent_id="agent-1",
                    message="Search for SafeHarness.",
                )
            )
        )
        pending = None
        for _ in range(100):
            pending = confirmation.get_pending("chat-approved")
            if pending is not None:
                break
            await asyncio.sleep(0.01)
        assert pending is not None
        approved = confirmation.set_decision(
            "chat-approved",
            pending["confirm_id"],
            confirmation.ALLOW,
            approver="user-1",
        )
        assert approved["ok"] is True
        return await task

    result = asyncio.run(run())
    assert len(calls) == 1
    assert result.blocked is False
    assert any(
        event["type"] == "approval" and event["data"]["status"] == "approved"
        for event in result.events
    )
    assert any(event["type"] == "execution" for event in result.events)
    assert any(event["type"] == "receipt" for event in result.events)
