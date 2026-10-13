"""SafeHarness guard for the main-chain MCP channel (additive, env-gated).

The marketplace / built-in MCP tools are assembled inside agentscope's Toolkit
and historically bypassed SafeHarness entirely: no policy evaluation, no
descriptor scanning. This module closes both gaps with two env-gated pieces
(default OFF — with the envs unset the agent builds exactly as before):

- ``McpDescriptorGuardClient``: ``BareNameMCPClient`` subclass whose
  ``list_tools`` runs every tool's name/description through the SafeHarness
  injection-pattern scan (the same patterns that guard gateway-registered
  descriptors). Warn-only by default; ``SAFEHARNESS_MCP_DESCRIPTOR_ENFORCE``
  removes flagged tools from the surface (fail-closed, opt-in).
- ``SafeHarnessToolPolicyMiddleware``: an ``on_acting`` middleware (same
  interception shape as OntologyGateMiddleware) that evaluates every tool call
  through the platform SafeHarness policy tuple. On ``TOOL_CALL_PROPOSED`` the
  decisive layers are SqlGuard and the trajectory policies (chain rules —
  including M5 tag chains, repeated attempts, entropy payloads); InformPolicy
  does not subscribe to tool proposals (it guards content and output surfaces),
  so the middleware keeps it in the tuple for future-proofing but it abstains
  here. Installed only when ``SAFEHARNESS_MCP_POLICY_GUARD`` is enabled.

Out of scope by design: capability signing / enforcement receipts (the gateway
process holds a static tool registry while MCP tools are per-tenant dynamic).
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, List

from agentscope.message import TextBlock, ToolResultState
from agentscope.middleware import MiddlewareBase
from agentscope.tool._response import ToolChunk

from core.llm.mcp_manager import BareNameMCPClient

logger = logging.getLogger(__name__)

ENV_DESCRIPTOR_ENFORCE = "SAFEHARNESS_MCP_DESCRIPTOR_ENFORCE"
ENV_POLICY_GUARD = "SAFEHARNESS_MCP_POLICY_GUARD"


def _env_flag(environ, name: str) -> bool:
    return (environ.get(name) or "").strip().lower() in ("1", "true", "yes")


def descriptor_guard_enabled(environ=None) -> bool:
    """True when flagged MCP tools must be removed from the surface."""
    import os

    return _env_flag(os.environ if environ is None else environ, ENV_DESCRIPTOR_ENFORCE)


def policy_guard_enabled(environ=None) -> bool:
    """True when the SafeHarness tool-policy middleware should be installed."""
    import os

    return _env_flag(os.environ if environ is None else environ, ENV_POLICY_GUARD)


def apply_descriptor_guard(tools: List[Any], *, server: str = "") -> List[Any]:
    """Scan every listed tool's name/description; drop flagged ones in enforce mode.

    Warn-only by default — the returned list equals the input unless
    ``SAFEHARNESS_MCP_DESCRIPTOR_ENFORCE`` is enabled and a tool is flagged.
    """
    kept: List[Any] = []
    for tool in tools:
        fields = _flagged_fields(getattr(tool, "name", ""), getattr(tool, "description", ""))
        if not fields:
            kept.append(tool)
            continue
        if descriptor_guard_enabled():
            logger.warning(
                "[mcp-guard] descriptor enforce: dropping tool '%s' of server '%s' (flagged fields: %s)",
                getattr(tool, "name", ""), server, fields,
            )
            continue
        logger.warning(
            "[mcp-guard] descriptor scan flagged tool '%s' of server '%s' (fields: %s); "
            "warn-only — set %s=1 to enforce",
            getattr(tool, "name", ""), server, fields, ENV_DESCRIPTOR_ENFORCE,
        )
        kept.append(tool)
    return kept


class McpDescriptorGuardClient(BareNameMCPClient):
    """``BareNameMCPClient`` whose listed tools pass the injection scan.

    Scanning happens in ``list_tools`` so every enumeration (schema build and
    dispatch alike) is covered, for stdio-pooled and per-request HTTP clients
    alike. Enforce mode drops flagged tools instead of only warning.
    """

    async def list_tools(self) -> List[Any]:
        return apply_descriptor_guard(await super().list_tools(), server=self.name)


_SCAN_POLICY = None


def _flagged_fields(name: Any, description: Any) -> List[str]:
    """Injection-pattern fields found in a tool's name/description (scan-only)."""
    global _SCAN_POLICY
    if _SCAN_POLICY is None:
        from safeharness.policies.inform import InformPolicy

        _SCAN_POLICY = InformPolicy()
    fields: List[str] = []
    for field, value in (("name", name), ("description", description)):
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            _, matched, _ = _SCAN_POLICY._sanitize_value(value)
        except Exception:  # noqa: BLE001 — scan failure must never break listing
            continue
        if matched:
            fields.append(field)
    return fields


class SafeHarnessToolPolicyMiddleware(MiddlewareBase):
    """Evaluate every tool call through the platform SafeHarness policy tuple.

    Interception shape copies ``OntologyGateMiddleware``: deny by yielding a
    DENIED ``ToolChunk`` without calling ``next_handler``. Policy-engine
    failures deny as well (fail-closed, same contract as the framework engine).
    The evaluation state lives on this instance; one middleware is created per
    agent (per request), so trajectory pressure is naturally run-scoped.
    """

    def __init__(self) -> None:
        from safeharness.policies.engine import PolicyEngine

        from core.safeharness_inform import (
            platform_extra_policies,
            platform_inform_policy,
        )
        from safeharness.policies.sql_guard import SqlGuardPolicy

        self._engine = PolicyEngine(
            (platform_inform_policy(), SqlGuardPolicy(), *platform_extra_policies())
        )
        self._state: dict = {}
        self._sequence = 0

    async def on_acting(self, agent, input_kwargs: dict, next_handler):  # noqa: ANN001
        tool_call = input_kwargs.get("tool_call")
        tool_name = str(getattr(tool_call, "name", "") or "")
        raw_input = getattr(tool_call, "input", "{}")
        try:
            tool_input = raw_input if isinstance(raw_input, dict) else json.loads(raw_input or "{}")
        except (TypeError, json.JSONDecodeError):
            tool_input = {}
        if not isinstance(tool_input, dict):
            tool_input = {}

        decision = await self._evaluate(agent, tool_name, tool_input)
        if decision is not None:
            payload = {
                "code": "safeharness_tool_policy_denied",
                "tool": tool_name,
                "reason_code": decision.code,
                "reason": decision.message,
            }
            logger.warning(
                "[mcp-guard] policy denied tool '%s': %s (%s)",
                tool_name, decision.code, decision.message,
            )
            yield ToolChunk(
                content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
                state=ToolResultState.DENIED,
                metadata={"safeharness_guard": payload},
            )
            return

        async for item in next_handler(**input_kwargs):
            yield item

    async def _evaluate(self, agent, tool_name: str, tool_input: dict):
        """Run the policy engine; None = allowed/abstained, Reason = deny."""
        from safeharness.api.decisions import Reason
        from safeharness.api.events import EventKind, EventMeta, SecurityEvent
        from safeharness.policies.base import PolicyContext

        self._sequence += 1
        meta = EventMeta(
            tenant_id=str(getattr(agent.state, "user_id", "") or "sumc"),
            session_id=str(getattr(agent.state, "session_id", "") or "unknown"),
            run_id=str(getattr(agent.state, "run_id", "") or "unknown"),
            agent_id=str(getattr(agent, "name", "") or "main_agent"),
            sequence=self._sequence,
            framework="agentscope",
            trace_id=str(getattr(agent.state, "trace_id", "") or ""),
        )
        event = SecurityEvent(
            EventKind.TOOL_CALL_PROPOSED,
            meta,
            {"tool_name": tool_name, "arguments": dict(tool_input)},
        )
        context = PolicyContext(session=None, state=self._state, services=None)
        try:
            decision = await self._engine.evaluate(event, context)
        except Exception as exc:  # noqa: BLE001 — framework-level failure: deny
            logger.warning("[mcp-guard] policy engine failure: %s", exc, exc_info=True)
            return Reason(
                code="safeharness_guard_engine_failure",
                message=f"策略引擎异常，默认拦截：{exc}",
                policy_id="safeharness.mcp_guard",
            )
        if decision is None or not decision.blocked:
            return None
        reasons = decision.reasons or []
        return reasons[-1]


__all__ = [
    "ENV_DESCRIPTOR_ENFORCE",
    "ENV_POLICY_GUARD",
    "McpDescriptorGuardClient",
    "SafeHarnessToolPolicyMiddleware",
    "descriptor_guard_enabled",
    "policy_guard_enabled",
]
