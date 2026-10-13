"""Optional SafeHarness protected-chat deployment bootstrap."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from core.config.settings import settings

logger = logging.getLogger(__name__)


def build_gateway_routing(gateway_client):
    """Issuer + executor pair that routes capability issuance and tool
    execution through the central gateway while the local runtime keeps the
    policy decision and the enforcement receipt.

    The issuer captures the live session context and the issued grant so the
    executor can bind them: the runtime's around_tool rebuilds the invocation
    from its event payload, where the capability token is absent — the token
    therefore rides on this closure, not on the executor's argument.
    Returns ``None`` when no gateway deployment is configured.
    """
    if gateway_client is None:
        return None
    from dataclasses import replace

    box: dict[str, Any] = {}

    class _GatewayIssuer:
        async def issue(self, context, descriptor, invocation):
            box["context"] = context
            grant = await gateway_client.issue(context, descriptor, invocation)
            box["token"] = grant.token
            return grant

    async def _gateway_executor(invocation):
        context = box.get("context")
        token = box.get("token")
        if context is None or not token:
            raise RuntimeError("gateway execution before capability issuance")
        return await gateway_client.execute(
            context, replace(invocation, capability_token=token)
        )

    return _GatewayIssuer(), _gateway_executor


def build_gateway_authority(gateway_client):
    """Policy-side capability authority backed by the remote gateway.

    ``ConstrainPolicy`` verifies every tool call against
    ``runtime.services.capability_authority``; when issuance is routed to the
    gateway, that authority must verify gateway-signed tokens — this wrapper
    delegates ``authorize`` to the gateway's non-consuming inspect (network or
    protocol failures deny by default, preserving fail-closed semantics).
    Returns ``None`` when no gateway deployment is configured.
    """
    if gateway_client is None:
        return None

    class _GatewayAuthority:
        async def issue(self, context, descriptor, invocation):
            return await gateway_client.issue(context, descriptor, invocation)

        async def authorize(self, session, descriptor, invocation):
            return await gateway_client.authorize(
                getattr(session, "context", session), descriptor, invocation
            )

    return _GatewayAuthority()


def bootstrap_protected_chat() -> Any | None:
    """Register the optional LangGraph adapter, or preserve route-level 503."""

    if not settings.safeharness_protected_chat.enabled:
        return None

    try:
        from api.routes.v1.safeharness import register_protected_chat_stream
        from core.llm.providers.registry import get_spec
        from core.services.model_config import ModelConfigService
        from langchain_openai import ChatOpenAI
        from safeharness import SafeHarnessRuntime
        from safeharness.adapters.langgraph import LangGraphAgent
        from safeharness.chat_bridge import ProtectedChatRequest, ProtectedChatRunner
        from core.safeharness_capability_mapping import build_internet_search_tool, build_local_runtime
        from core.safeharness_approval import SUMCApprovalBroker
        from core.safeharness_dispatch import (
            classify_protected_message,
            detect_dangerous_sql,
            detect_prompt_injection,
        )
    except ImportError as exc:
        logger.warning("[safeharness] protected chat disabled: missing dependency: %s", exc)
        return None

    runtimes: dict[int, Any] = {}
    judges: dict[str, Any] = {}

    def runtime_key(request: ProtectedChatRequest) -> int:
        return id(request)

    def runtime_for(request: ProtectedChatRequest):
        tool = build_internet_search_tool()
        runtime_options = {
            "agent_id": request.agent_id,
            "approval_broker": SUMCApprovalBroker(),
        }
        judge = judges.get(str(getattr(request, "run_id", "")))
        if judge is not None:
            runtime_options["judge_provider"] = judge
        # Gateway routing: when configured, the policy-side capability authority
        # must verify gateway-signed tokens (ConstrainPolicy runs before the
        # executor), so the local random-key authority is replaced.
        from core.safeharness_tool_gateway import gateway_client_from_environment

        gateway_client = gateway_client_from_environment()
        if gateway_client is not None:
            gateway_authority = build_gateway_authority(gateway_client)
            if gateway_authority is not None:
                runtime_options["capability_authority"] = gateway_authority
        runtime = build_local_runtime(tool, **runtime_options)
        runtimes[runtime_key(request)] = runtime
        return runtime

    def valid_model_config(current):
        if current is None or not current.base_url or not current.api_key or not current.model_name:
            return False
        current_provider = get_spec(current.provider)
        return current_provider.engine == "openai" and current_provider.id != "azure_openai"

    def resolve_model_config(provider_id: str = ""):
        service = ModelConfigService.get_instance()
        current = service.resolve_provider(provider_id) if provider_id else service.resolve("main_agent")
        if not valid_model_config(current):
            raise RuntimeError("main_agent model configuration is unavailable")
        return current

    def resolve_judge_config(provider_id: str):
        service = ModelConfigService.get_instance()
        candidates = (
            service.resolve("judge"),
            service.resolve_provider(provider_id) if provider_id else None,
            service.resolve("main_agent"),
        )
        for current in candidates:
            try:
                if valid_model_config(current):
                    return current
            except Exception:
                continue
        raise RuntimeError("judge model configuration is unavailable")

    def model_from_config(current):
        return ChatOpenAI(
            api_key=current.api_key,
            base_url=current.base_url,
            model=current.model_name,
            temperature=current.temperature,
            max_tokens=current.max_tokens,
            timeout=current.timeout,
        )

    def resolve_model(provider_id: str):
        return model_from_config(resolve_model_config(provider_id))

    class CountingJudge:
        def __init__(self, provider, model_name: str) -> None:
            self.provider = provider
            self.model = model_name
            self.calls = 0
            self.last_status = "not_called"
            self._cache: dict[str, Any] = {}

        async def judge(self, request: JudgeRequest):
            key = json.dumps(request.content, ensure_ascii=False, sort_keys=True, default=str)
            if key not in self._cache:
                self.calls += 1
                try:
                    self._cache[key] = await self.provider.judge(request)
                except Exception:
                    self.last_status = "ERROR"
                    raise
            result = self._cache[key]
            self.last_status = result.label.strip().upper() or "UNCERTAIN"
            return result

    def resolve_judge(provider_id: str) -> CountingJudge:
        from safeharness.backends.openai_judge import OpenAICompatibleJudge

        config = resolve_judge_config(provider_id)
        model = model_from_config(config)
        judge_model = model.bind(response_format={"type": "json_object"})

        def caller(messages, _model_name):
            response = judge_model.invoke(list(messages))
            content = getattr(response, "content", "")
            return content if isinstance(content, str) else str(content or "")

        provider = OpenAICompatibleJudge(caller, model=config.model_name)
        return CountingJudge(provider, config.model_name)

    async def inspect_with_judge(context: Mapping[str, str], judge: CountingJudge):
        from core.safeharness_audit import build_audit_sink
        from safeharness.api.context import SessionContext
        from core.safeharness_inform import platform_inform_and_extras
        from safeharness.runtime import RuntimeConfig, RuntimeServices

        # The input decision is the security boundary itself — persist it to the
        # audit sink so allow/deny outcomes survive the process.
        runtime = SafeHarnessRuntime(
            policies=platform_inform_and_extras(),
            services=RuntimeServices(judge_provider=judge, audit_sink=build_audit_sink()),
            config=RuntimeConfig(policy_version="safeharness.default.v1"),
        )
        session = runtime.session(SessionContext(
            tenant_id=context["tenant_id"],
            session_id=context["session_id"],
            run_id=context["run_id"],
            agent_id=context["agent_id"],
            trace_id=context["trace_id"],
            framework="sumc-protected-chat",
        ))
        async with session:
            return await session.inspect_content(context["message"])

    async def setup_noc_delegation(context: Mapping[str, str]) -> None:
        gateway_url = os.environ.get("SAFEHARNESS_COMMUNICATION_GATEWAY_URL", "").rstrip("/")
        admin_key = os.environ.get("SAFEHARNESS_COMMUNICATION_ADMIN_API_KEY", "")
        if not gateway_url or not admin_key:
            raise RuntimeError("communication administration is unavailable")
        # The gateway registers NOC principals under the communication tenant,
        # not under the per-request platform tenant, so delegation administration
        # must address the same tenant the NOC chain itself runs in.
        tenant_id = (
            os.environ.get("SAFEHARNESS_COMMUNICATION_TENANT_ID", "").strip()
            or context["tenant_id"]
        )
        payload = json.dumps({
            "tenant_id": tenant_id,
            "grantor_agent_id": "noc-supervisor-agent",
            "grantee_agent_id": "noc-diagnosis-agent",
            "capabilities": ["nms:alarms:read"],
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            "session_id": context["session_id"],
            "run_id": context["run_id"],
            "max_messages": 1,
            "delegation_id": context["delegation_id"],
        }).encode("utf-8")

        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        def send() -> None:
            request = Request(
                f"{gateway_url}/v1/admin/delegations",
                data=payload,
                headers={"Authorization": f"Bearer {admin_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with build_opener(NoRedirect).open(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError("delegation administration rejected the request")

        await asyncio.to_thread(send)

    def agent_factory(request: ProtectedChatRequest) -> LangGraphAgent:
        runtime = runtimes.pop(runtime_key(request), None)
        if runtime is None:
            raise RuntimeError("protected chat runtime is unavailable")
        # Resolve per run so operator model changes are not pinned at process startup.
        provider_id = str(getattr(request, "metadata", {}).get("model_provider_id") or "")
        model = resolve_model(provider_id)
        tools = (build_internet_search_tool(),)
        capability_issuer = runtime.services.capability_authority

        # Central-gateway routing (env-driven; absent URL keeps local execution):
        # capability issuance goes to the gateway and executions are dispatched
        # through it, while the local runtime still makes the policy decision and
        # countersigns the enforcement receipt (the gateway verifies both proofs).
        # Requires the session tenant/agent pair to equal the gateway's
        # capability-mapping principal (true for this deployment topology).
        from dataclasses import replace

        from core.safeharness_tool_gateway import gateway_client_from_environment

        gateway_client = gateway_client_from_environment()
        routing = build_gateway_routing(gateway_client)
        if routing is not None:
            issuer, executor = routing
            # executor=None is required: LangGraphTool allows exactly one of the
            # local executor and the gateway invocation executor.
            tools = (replace(tools[0], executor=None, invocation_executor=executor),)
            capability_issuer = issuer

        return LangGraphAgent(
            model=model,
            tools=tools,
            capability_issuer=capability_issuer,
            system_prompt=settings.safeharness_protected_chat.system_prompt,
        )

    runner = ProtectedChatRunner(
        agent_factory,
        runtime_factory=runtime_for,
        security_timeline=True,
    )
    base_stream = runner.astream_context

    def event(context: Mapping[str, str], sequence: int, event_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_id": str(uuid4()),
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
            "data": dict(data),
        }

    def decorate(raw: Mapping[str, Any], decision) -> dict[str, Any]:
        result = dict(raw)
        data = dict(result.get("data") or {})
        route = " -> ".join(decision.route)
        if result.get("type") in {"policy_decision", "execution"}:
            data.update({"route": route, "agent_id": decision.terminal_agent_id})
            if result.get("type") == "execution" and data.get("status") != "dispatched":
                data.setdefault("stage", "tool")
        elif result.get("type") == "run_finished":
            data.update({"route": route, "agent_id": decision.terminal_agent_id, "agent_chain": route})
        elif result.get("type") == "blocked":
            if str(data.get("reason_code") or "").startswith("prompt_injection"):
                data.pop("route", None)
                data.pop("agent_chain", None)
            else:
                data.update({"route": route, "agent_chain": route})
        result["data"] = data
        return result

    async def knowledge_unavailable(context: Mapping[str, str], judge):
        yield event(context, 1, "run_started", {})
        yield event(
            context,
            2,
            "error",
            {
                "code": "knowledge_qa_unavailable",
                "message": "知识问答 Agent 当前未接入受保护链。",
                "judge_status": judge.last_status,
                "judge_calls": judge.calls,
                "judge_model": judge.model,
            },
        )
        yield event(context, 3, "run_finished", {"status": "error", "route": " -> ".join(("platform-supervisor-agent", "knowledge-qa-agent")), "agent_id": "knowledge-qa-agent", "agent_chain": "platform-supervisor-agent -> knowledge-qa-agent", "model_calls": 0, "tool_calls_proposed": 0, "tool_calls_started": 0, "judge_status": judge.last_status, "judge_calls": judge.calls, "judge_model": judge.model})

    async def input_blocked(context: Mapping[str, str], reason_code: str, judge=None):
        judge_status = judge.last_status if judge else "not_called"
        judge_calls = judge.calls if judge else 0
        judge_model = judge.model if judge else ""
        judge_evidence = {"judge_status": judge_status, "judge_calls": judge_calls}
        if judge_model:
            judge_evidence["judge_model"] = judge_model
        message = "Judge 未能安全确认，已阻断。" if reason_code == "semantic_prompt_injection_uncertain" else "SafeHarness 已阻断本轮请求。"
        yield event(context, 1, "run_started", {})
        yield event(
            context,
            2,
            "policy_decision",
            {
                "decision_id": "safeharness-input-guard",
                "event_kind": "content_received",
                "disposition": "deny",
                "policy_version": "safeharness.default.v1",
                "reason_code": reason_code,
                "stage": "input_guard",
                **judge_evidence,
            },
        )
        yield event(
            context,
            3,
            "blocked",
            {
                "reason_code": reason_code,
                "message": message,
                "source": "content_received",
                "action": "INPUT_BLOCKED; MODEL_SKIPPED",
                "model_calls": 0,
                "tool_calls_proposed": 0,
                "tool_calls_started": 0,
                "final_status": "blocked",
                **judge_evidence,
            },
        )
        yield event(
            context,
            4,
            "run_finished",
            {
                "status": "blocked",
                "model_calls": 0,
                "tool_calls_proposed": 0,
                "tool_calls_started": 0,
                **judge_evidence,
            },
        )

    async def noc_stream(context: Mapping[str, str], decision, judge):
        from examples.langgraph_noc.multi_agent import run_configured_multi_agent

        sequence = 0

        def emit(event_type: str, data: Mapping[str, Any]):
            nonlocal sequence
            sequence += 1
            return event(context, sequence, event_type, data)

        yield emit("run_started", {})
        route = " -> ".join(decision.route)
        try:
            await setup_noc_delegation(context)
        except Exception:
            yield emit("error", {"code": "delegation_setup_failed", "message": "受保护运维授权创建失败，已停止本轮处理。"})
            yield emit("run_finished", {"status": "error", "route": route, "agent_id": decision.terminal_agent_id, "agent_chain": route, "model_calls": 0, "tool_calls_proposed": 0, "tool_calls_started": 0, "judge_status": judge.last_status, "judge_calls": judge.calls, "judge_model": judge.model})
            return
        yield emit(
            "policy_decision",
            {
                "decision_id": "platform-dispatch",
                "event_kind": "content_received",
                "disposition": "allow",
                "policy_version": "safeharness.default.v1",
                "reason_code": "content_accepted",
                "stage": "input_guard",
                "route": route,
                "agent_id": "platform-supervisor-agent",
            },
        )
        yield emit(
            "execution",
            {
                "decision_id": "platform-dispatch",
                "tool_name": "platform-supervisor-agent",
                "status": "dispatched",
                "stage": "dispatch",
                "route": route,
                "agent_id": "platform-supervisor-agent",
            },
        )
        try:
            result = await run_configured_multi_agent(
                context["message"],
                model=resolve_model(context.get("model_provider_id", "")),
                session_id=context["session_id"],
                run_id=context["run_id"],
                delegation_id=context.get("delegation_id") or context["run_id"],
                judge_provider=judge,
            )
            trace = result.get("security_trace") or {}
            summary = trace.get("summary") or {}
            entries = trace.get("entries") or []
            first = entries[0] if entries else {}
            reason_code = str(result.get("security_reason") or first.get("reason_code") or "content_accepted")
            for entry in entries:
                entry_type = str(entry.get("event", ""))
                if entry_type == "tool_call_proposed":
                    yield emit("tool_proposal", {"decision_id": "noc-tool", "tool_name": str(entry.get("tool") or "read_only_diagnosis"), "route": route, "agent_id": "noc-diagnosis-agent"})
                elif entry_type == "tool_call_started":
                    yield emit("execution", {"decision_id": "noc-tool", "tool_name": str(entry.get("tool") or "read_only_diagnosis"), "status": "started", "stage": "tool", "route": route, "agent_id": "noc-diagnosis-agent"})
                elif entry_type in {"tool_call_succeeded", "tool_call_failed"}:
                    yield emit("execution", {"decision_id": "noc-tool", "tool_name": str(entry.get("tool") or "read_only_diagnosis"), "status": "succeeded" if entry_type == "tool_call_succeeded" else "failed", "stage": "tool", "route": route, "agent_id": "noc-diagnosis-agent"})
            model_calls = int(summary.get("model_calls", 0) or 0)
            tool_calls_proposed = int(summary.get("tool_calls_proposed", 0) or 0)
            tool_calls_started = int(summary.get("tool_calls_started", 0) or 0)
            if result.get("halted"):
                action = str(first.get("action") or "INPUT_BLOCKED; MODEL_SKIPPED")
                yield emit("blocked", {"reason_code": reason_code, "message": "SafeHarness 已阻断受保护运维通信。", "source": "inter_agent", "action": action, "route": route, "agent_chain": route, "model_calls": model_calls, "tool_calls_proposed": tool_calls_proposed, "tool_calls_started": tool_calls_started, "final_status": "blocked", "judge_status": judge.last_status, "judge_calls": judge.calls, "judge_model": judge.model})
                yield emit("run_finished", {"status": "blocked", "route": route, "agent_id": decision.terminal_agent_id, "agent_chain": route, "model_calls": model_calls, "tool_calls_proposed": tool_calls_proposed, "tool_calls_started": tool_calls_started, "judge_status": judge.last_status, "judge_calls": judge.calls, "judge_model": judge.model})
                return
            response = str(result.get("final_response") or "")
            if not response:
                raise RuntimeError("NOC diagnosis returned no response")
            yield emit("message", {"content": response})
            yield emit("run_finished", {"status": "completed", "route": route, "agent_id": decision.terminal_agent_id, "agent_chain": route, "model_calls": model_calls, "tool_calls_proposed": tool_calls_proposed, "tool_calls_started": tool_calls_started, "judge_status": judge.last_status, "judge_calls": judge.calls, "judge_model": judge.model})
        except Exception:
            yield emit("error", {"code": "protected_noc_failed", "message": "受保护运维 Agent 暂时不可用。"})
            yield emit("run_finished", {"status": "error", "route": route, "agent_id": decision.terminal_agent_id, "agent_chain": route, "model_calls": 0, "tool_calls_proposed": 0, "tool_calls_started": 0, "judge_status": judge.last_status, "judge_calls": judge.calls, "judge_model": judge.model})

    async def dispatch_stream(context: Mapping[str, str]):
        reason_code = detect_prompt_injection(context["message"]) or detect_dangerous_sql(
            context["message"]
        )
        if reason_code:
            # Structural blocks skip the SafeHarness session; write the audit
            # record explicitly so denials stay reviewable after a restart.
            from core.safeharness_audit import record_blocked_input

            await record_blocked_input(context, reason_code)
            async for item in input_blocked(context, reason_code):
                yield item
            return
        try:
            judge = resolve_judge(context.get("model_provider_id", ""))
            decision = await inspect_with_judge(context, judge)
        except Exception:
            judge = locals().get("judge")
            if judge is None:
                judge = type("UnavailableJudge", (), {"last_status": "ERROR", "calls": 0, "model": "unavailable"})()
            async for item in input_blocked(context, "semantic_prompt_injection_uncertain", judge):
                yield item
            return
        if not decision.executable:
            reason = decision.reasons[0].code if decision.reasons else "semantic_prompt_injection_uncertain"
            async for item in input_blocked(context, reason, judge):
                yield item
            return
        judges[context["run_id"]] = judge
        decision = classify_protected_message(context["message"])
        if decision.target == "noc":
            async for item in noc_stream(context, decision, judge):
                yield item
            judges.pop(context["run_id"], None)
            return
        if decision.target == "knowledge_qa":
            async for item in knowledge_unavailable(context, judge):
                yield item
            judges.pop(context["run_id"], None)
            return
        counters = {"model_calls": 0, "tool_calls_proposed": 0, "tool_calls_started": 0}
        async for item in base_stream(context):
            event_type = str(item.get("type") or "")
            data = item.get("data") if isinstance(item.get("data"), Mapping) else {}
            if event_type == "tool_proposal":
                counters["tool_calls_proposed"] += 1
            elif event_type == "execution" and str(data.get("status") or "") == "started":
                counters["tool_calls_started"] += 1
            elif event_type == "run_finished":
                counters["model_calls"] = 1 if str(data.get("status") or "") == "completed" else 0
                item = dict(item)
                item["data"] = {**dict(data), **counters, "judge_status": judge.last_status, "judge_calls": judge.calls, "judge_model": judge.model}
            yield decorate(item, decision)
        judges.pop(context["run_id"], None)

    # Keep runner.astream_context as the registered callable for existing
    # integrations while routing its implementation through the dispatcher.
    runner.astream_context = dispatch_stream
    register_protected_chat_stream(runner.astream_context, replace=True)
    logger.info("[safeharness] protected chat stream registered")
    return runner


def shutdown_protected_chat() -> None:
    """Remove the process-local handler during app shutdown/reload."""

    try:
        from api.routes.v1.safeharness import unregister_protected_chat_stream
    except ImportError:
        return
    unregister_protected_chat_stream()
