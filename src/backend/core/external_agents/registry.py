"""External agent registry — catalog, availability probe, guarded dispatch.

Dispatch order is the security boundary: SafeHarness structural checks run
*before* any framework sees the message, and a block is audited exactly like a
protected-chat block. A framework that is not installed reports as unavailable
instead of failing the whole endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from core.external_agents.runners import RUNNERS, RunnerUnavailable, RunnerResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalAgentCard:
    agent_id: str
    name: str
    framework: str
    project: str
    description: str
    runner: Any  # coroutine function

    @property
    def public(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "framework": self.framework,
            "project": self.project,
            "description": self.description,
        }


def _card(agent_id: str, name: str, framework: str, project: str, description: str) -> ExternalAgentCard:
    return ExternalAgentCard(
        agent_id=agent_id,
        name=name,
        framework=framework,
        project=project,
        description=description,
        runner=RUNNERS[framework],
    )


# Catalog: general-purpose agents from real, maintained GitHub projects.
CATALOG: dict[str, ExternalAgentCard] = {
    card.agent_id: card
    for card in (
        _card(
            "ext-smolagents",
            "SmolAgents 代码智能体",
            "smolagents",
            "huggingface/smolagents",
            "HuggingFace CodeAgent：以生成并执行 Python 代码片完成任务的通用智能体。",
        ),
        _card(
            "ext-openai-agents",
            "OpenAI Agents 通用助手",
            "openai-agents",
            "openai/openai-agents-python",
            "OpenAI Agents SDK 的 ReAct 式通用助手智能体。",
        ),
        _card(
            "ext-autogen",
            "AutoGen 通用助手",
            "autogen",
            "microsoft/autogen",
            "Microsoft AutoGen AssistantAgent：事件驱动的通用对话智能体。",
        ),
    )
}


def available_external_agents() -> dict[str, Any]:
    """Catalog + per-framework availability (import probe, no side effects)."""
    import importlib.util

    modules = {
        "smolagents": "smolagents",
        "openai-agents": "agents",
        "autogen": "autogen_agentchat",
    }
    items = []
    for card in CATALOG.values():
        module = modules.get(card.framework, "")
        installed = importlib.util.find_spec(module) is not None if module else False
        items.append({**card.public, "available": installed})
    return {"agents": items}


def _model_provider_config(agent_id: str) -> dict[str, Any]:
    """The configured model for one agent: override → shared default → main.

    No new credentials are introduced — every candidate is an existing row of
    the platform's model-provider table.
    """
    from core.db.engine import SessionLocal
    from core.external_agents.model_settings import resolve_provider_chain
    from core.services.model_config import ModelConfigService

    with SessionLocal() as db:
        provider_id, source = resolve_provider_chain(db, agent_id)
    service = ModelConfigService.get_instance()
    if provider_id:
        current = service.resolve_provider(provider_id)
    else:
        current = service.resolve("main_agent")
        source = "main_agent"
    if current is None or not current.api_key or not current.base_url:
        raise RunnerUnavailable("main_agent model provider is unavailable")
    return {
        "base_url": current.base_url,
        "api_key": current.api_key,
        "model": current.model_name,
        "provider_id": provider_id or "main_agent",
        "source": source,
    }


class ExternalAgentBlocked(RuntimeError):
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


async def run_external_agent(
    agent_id: str,
    message: str,
    *,
    context: dict[str, str] | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Guarded dispatch: SafeHarness structural check → framework run → audit."""
    card = CATALOG.get(agent_id)
    if card is None:
        raise KeyError(agent_id)
    context = context or {}
    run_id = context.get("run_id") or f"ext-{uuid4().hex[:12]}"
    context = {**context, "run_id": run_id, "agent_id": card.agent_id}

    from core.safeharness_audit import record_blocked_input
    from core.safeharness_dispatch import detect_dangerous_sql, detect_prompt_injection

    reason = detect_prompt_injection(message) or detect_dangerous_sql(message)
    if reason:
        await record_blocked_input(context, reason, source="external_agent_input")
        raise ExternalAgentBlocked(reason)

    config = _model_provider_config(agent_id)
    from core.db.engine import SessionLocal
    from core.external_agents.supervisor import (
        SupervisionState,
        load_supervision_config,
        supervision_notice,
    )

    with SessionLocal() as db:
        supervision = SupervisionState(config=load_supervision_config(db))
        from core.external_agents.memory_settings import load_memory_config

        memory_config = load_memory_config(db)
    config["supervision"] = supervision
    # Session window: configurable, clamped at parse time.
    window = (history or [])[-memory_config.history_window:]
    config["history"] = window
    # Cross-session long-term memory (mem0): read-only injection before the run,
    # best-effort — a degraded retrieval contributes nothing and never fails the run.
    user_id = str(context.get("user_id") or "")
    long_term_status = "disabled"
    long_term_text = ""
    if memory_config.long_term_enabled and user_id:
        try:
            from core.config.settings import settings
            from core.memory.service import retrieve_memories

            if settings.memory.enabled:
                raw = await retrieve_memories(
                    user_id,
                    message,
                    limit=memory_config.long_term_limit,
                )
                long_term_text = (raw or "").strip()[: memory_config.long_term_max_chars]
                long_term_status = "retrieved" if long_term_text else "empty"
            else:
                long_term_status = "platform_disabled"
        except Exception as exc:  # noqa: BLE001 — memory is best-effort
            logger.warning("[external-agents] long-term memory read failed: %s", type(exc).__name__)
            long_term_status = "error"
    if long_term_text:
        config["long_term"] = long_term_text
    # Gateway-routed tool surface: mounted only when the deployment is fully
    # configured (gateway URL + receipt signing material) and reachable. The
    # gateway principal comes from the gateway's own environment defaults.
    from core.external_agents.gateway_tools import build_gateway_tool_bundle, tool_snapshot

    config["gateway_tool"] = await build_gateway_tool_bundle(
        session_id=run_id,
        run_id=run_id,
    )
    started = time.monotonic()
    try:
        result: RunnerResult = await card.runner(message, config)
    except RunnerUnavailable:
        raise
    duration_ms = int((time.monotonic() - started) * 1000)

    reply = result.reply
    review_status = "not_reviewed"
    if result.interventions:
        # Graded escalation: review whatever the run produced before degrading
        # the reply to the structured notice (fail-closed on judge errors).
        from core.evolution.content_gate import semantic_review

        review_target = reply or json.dumps(result.interventions[-1], ensure_ascii=False)
        gate = await semantic_review(review_target, context=context)
        review_status = f"{gate.judge_status}:{gate.reason_code}"
        reply = supervision_notice(result.interventions[-1], reviewed=gate.reason_code)
        await _record_supervision(context, card, result, gate)

    # Long-term memory write-back: only genuine (non-intervened) replies are
    # persisted, fire-and-forget, budgeted and gated like the read path.
    long_term_write = "skipped"
    if (
        memory_config.long_term_enabled
        and user_id
        and long_term_status in ("retrieved", "empty")
        and reply
        and not result.halted
    ):
        try:
            from core.config.settings import settings
            from core.memory.service import save_conversation

            if settings.memory.enabled:
                asyncio.create_task(save_conversation(user_id, message, reply))
                long_term_write = "scheduled"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[external-agents] long-term memory write skipped: %s", type(exc).__name__)
            long_term_write = "error"

    await _record_run(context, card, result)
    tool_calls = tool_snapshot(config.get("gateway_tool"))
    return {
        "reply": reply,
        "evidence": {
            "agent_id": card.agent_id,
            "framework": result.framework,
            "project": card.project,
            "model": result.model,
            "provider_id": config.get("provider_id"),
            "model_source": config.get("source"),
            "guard": "passed",
            "reason_code": None,
            "steps": result.steps,
            "memory_turns": len(window),
            "memory": {
                "window_turns": len(window),
                "window_configured": memory_config.history_window,
                "long_term_read": long_term_status,
                "long_term_chars": len(long_term_text),
                "long_term_write": long_term_write,
            },
            "tools": tool_calls,
            "supervision": {
                "interventions": result.interventions,
                "halted": result.halted,
                "review": review_status,
            } if result.interventions else {"interventions": [], "halted": False},
            "duration_ms": result.duration_ms or duration_ms,
            "run_id": run_id,
        },
    }


async def _record_supervision(
    context: dict[str, str],
    card: ExternalAgentCard,
    result: RunnerResult,
    gate: Any,
) -> None:
    """Audit a supervision intervention (deny — the run was cut short)."""
    try:
        from safeharness.api.decisions import Decision, Disposition, Reason
        from safeharness.api.events import EventKind, EventMeta, SecurityEvent
        from safeharness.ports import AuditRecord

        from core.safeharness_audit import build_audit_sink

        intervention = result.interventions[-1] if result.interventions else {}
        trigger = str(intervention.get("trigger") or "unknown")
        meta = EventMeta(
            tenant_id=str(context.get("tenant_id") or ""),
            session_id=str(context.get("session_id") or ""),
            run_id=str(context.get("run_id") or ""),
            agent_id=str(context.get("agent_id") or ""),
            sequence=1,
            framework="sumc-external-agents",
            trace_id=str(context.get("trace_id") or ""),
        )
        event = SecurityEvent(
            kind=EventKind.CONTENT_RECEIVED,
            meta=meta,
            payload={
                "source": "external_agent_supervision",
                "framework": card.framework,
                "trigger": trigger,
                "intervention": intervention,
                "review": f"{gate.judge_status}:{gate.reason_code}",
                "redacted": True,
            },
        )
        decision = Decision(
            disposition=Disposition.DENY,
            effective_event=event,
            reasons=(
                Reason(
                    f"supervision_{trigger}",
                    "Runtime supervisor intervened; the run was cut short.",
                    "safeharness.external-agents.supervision",
                ),
            ),
            policy_version="safeharness.default.v1",
        )
        await build_audit_sink().write(AuditRecord("decision", event, decision=decision))
    except Exception as exc:  # noqa: BLE001 — audit noise never breaks the run
        logger.warning("[external-agents] supervision audit write failed: %s", type(exc).__name__)


async def _record_run(context: dict[str, str], card: ExternalAgentCard, result: RunnerResult) -> None:
    """Audit an allowed external-agent run (mirror of the block record)."""
    try:
        from safeharness.api.decisions import Decision, Disposition, Reason
        from safeharness.api.events import EventKind, EventMeta, SecurityEvent
        from safeharness.ports import AuditRecord

        from core.safeharness_audit import build_audit_sink

        meta = EventMeta(
            tenant_id=str(context.get("tenant_id") or ""),
            session_id=str(context.get("session_id") or ""),
            run_id=str(context.get("run_id") or ""),
            agent_id=str(context.get("agent_id") or ""),
            sequence=1,
            framework="sumc-external-agents",
            trace_id=str(context.get("trace_id") or ""),
        )
        event = SecurityEvent(
            kind=EventKind.CONTENT_RECEIVED,
            meta=meta,
            payload={
                "source": "external_agent_input",
                "framework": card.framework,
                "model": result.model,
                "redacted": True,
            },
        )
        decision = Decision(
            disposition=Disposition.ALLOW,
            effective_event=event,
            reasons=(
                Reason(
                    "external_agent_run_allowed",
                    "Structural checks passed; executed by the external framework agent.",
                    "safeharness.inform.patterns",
                ),
            ),
            policy_version="safeharness.default.v1",
        )
        await build_audit_sink().write(AuditRecord("decision", event, decision=decision))
    except Exception as exc:  # noqa: BLE001 — audit noise never breaks the run
        logger.warning("[external-agents] run audit write failed: %s", type(exc).__name__)


__all__ = [
    "ExternalAgentBlocked",
    "available_external_agents",
    "run_external_agent",
]
