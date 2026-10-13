"""SafeHarness content gate for evolution materialisation.

The evolution chain (conversation → candidate → approval → materialisation →
load) is the one write path into the runtime that never went through a
SafeHarness checkpoint: the SKILL.md body is LLM-written from mined evidence,
and mined evidence is conversation-derived, so a prompt injection that survived
into stored memories can come back as instructions in a skill the agent later
reads. This gate closes that hole at the only choke point materialisation has.

Two layers, in cost order:

1. **Structural** — ``InformPolicy`` regex scan (fail-closed, zero model calls,
   millisecond). A hit blocks outright.
2. **Semantic** — the same ``prompt_injection_detection`` judge the protected
   chat chain uses (judge role → main_agent fallback), run through a
   SafeHarnessRuntime session so the decision lands in the audit ledger.
   Fail-closed: ``INJECTION``, ``UNCERTAIN``, an unavailable judge, or any
   error all block; only an explicit ``SAFE`` passes.

Materialisation is human-gated and rare, so the thread-per-call bridge below
(needed because the activation path is synchronous while the judge is async)
costs nothing worth optimising.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

GATE_FRAMEWORK = "sumc-evolution-activation"
_JUDGE_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class GateDecision:
    """Outcome of inspecting one piece of materialisation content."""

    allowed: bool
    reason_code: str
    detail: str = ""
    judge_status: str = "not_called"
    judge_model: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "judge_status": self.judge_status,
            "judge_model": self.judge_model,
        }


def _resolve_judge():
    """The configured semantic judge, or None when unresolvable.

    Same resolution order the protected chat chain uses: the dedicated judge
    role first, then the main agent. Returns (provider, model_name) or None.
    """
    from core.services.model_config import ModelConfigService
    from safeharness.backends.openai_judge import OpenAICompatibleJudge

    service = ModelConfigService.get_instance()
    config = None
    for candidate in (service.resolve("judge"), service.resolve("main_agent")):
        try:
            if candidate and valid_model_config(candidate):
                config = candidate
                break
        except Exception:  # noqa: BLE001 — an unusable role just falls through
            continue
    if config is None:
        return None

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )
    judge_model = model.bind(response_format={"type": "json_object"})

    def caller(messages, _model_name):
        response = judge_model.invoke(list(messages))
        content = getattr(response, "content", "")
        return content if isinstance(content, str) else str(content or "")

    return OpenAICompatibleJudge(caller, model=config.model_name), config.model_name


def valid_model_config(current) -> bool:
    """Whether a resolved model config can actually serve a request."""
    return bool(
        current
        and getattr(current, "api_key", None)
        and getattr(current, "model_name", None)
    )


async def _inspect_semantic(content: str, *, context: dict[str, str]) -> GateDecision:
    """The judge leg: SafeHarnessRuntime(InformPolicy) + inspect_content."""
    from safeharness import SafeHarnessRuntime
    from safeharness.api.context import SessionContext
    from core.safeharness_inform import platform_inform_policy
    from safeharness.runtime import RuntimeConfig, RuntimeServices

    from core.safeharness_audit import build_audit_sink

    resolved = _resolve_judge()
    if resolved is None:
        return GateDecision(
            allowed=False,
            reason_code="content_inspection_unavailable",
            detail="semantic judge 未配置（judge/main_agent 均不可用），fail-closed 拒绝",
            judge_status="unavailable",
        )
    provider, model_name = resolved

    runtime = SafeHarnessRuntime(
        policies=(platform_inform_policy(),),
        services=RuntimeServices(judge_provider=provider, audit_sink=build_audit_sink()),
        config=RuntimeConfig(policy_version="safeharness.default.v1"),
    )
    session = runtime.session(
        SessionContext(
            tenant_id=context.get("tenant_id") or "default",
            session_id=context.get("session_id") or "evolution-activation",
            run_id=context.get("run_id") or "evolution-activation",
            agent_id=context.get("agent_id") or "evolution-activation",
            trace_id=context.get("trace_id") or "evolution-activation",
            framework=GATE_FRAMEWORK,
        )
    )
    async with session:
        decision = await session.inspect_content(content)

    status = str(getattr(provider, "last_status", "") or "unknown")
    executable = bool(getattr(decision, "executable", False))
    reasons = list(getattr(decision, "reasons", None) or [])
    reason_code = str(reasons[0].code) if reasons and getattr(reasons[0], "code", None) else ""
    if executable:
        return GateDecision(
            allowed=True,
            reason_code=reason_code or "content_inspection_passed",
            judge_status=status,
            judge_model=model_name,
        )
    # Not executable covers INJECTION as well as UNCERTAIN/ERROR — the runtime
    # is fail-closed, and for this gate every non-pass is a block.
    return GateDecision(
        allowed=False,
        reason_code=reason_code or "semantic_prompt_injection_uncertain",
        detail="semantic judge 裁定为不可放行（fail-closed）",
        judge_status=status,
        judge_model=model_name,
    )


def _run_async_bridge(coro, *, timeout: float):
    """Run an async call from synchronous code, from any context.

    ``materialise_skill_candidate`` is sync and is reached both from async
    routes (running event loop — ``asyncio.run`` would refuse) and from admin
    scripts (no loop). A dedicated thread sidesteps both; materialisation is
    human-gated and rare, so the cost is irrelevant.
    """

    def _runner():
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller side
            outcome["error"] = exc

    outcome: dict[str, Any] = {}
    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"content inspection exceeded {timeout:.0f}s")
    if "error" in outcome:
        raise outcome["error"]
    if "value" not in outcome:
        raise RuntimeError("content inspection produced no result")
    return outcome["value"]


def inspect_materialisation_content(
    content: str,
    *,
    context: Optional[dict[str, str]] = None,
) -> GateDecision:
    """Structural + semantic injection gate for content about to be materialised.

    Raises nothing by design: every failure mode — including this gate's own —
    comes back as a blocking ``GateDecision``, because a gate that can fail
    open is not a gate.
    """
    text = str(content or "")
    if not text.strip():
        return GateDecision(
            allowed=False,
            reason_code="content_inspection_unavailable",
            detail="物化内容为空，拒绝",
        )

    from core.safeharness_dispatch import detect_prompt_injection

    structural = detect_prompt_injection(text)
    if structural:
        return GateDecision(
            allowed=False,
            reason_code=structural,
            detail="结构层命中注入模式（0 次 Judge 调用）",
            judge_status="not_called",
        )

    gate_context = dict(context or {})
    gate_context.setdefault("run_id", f"evo-gate-{json.dumps(text[:64], ensure_ascii=False)}")
    try:
        return _run_async_bridge(
            _inspect_semantic(text, context=gate_context),
            timeout=_JUDGE_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — fail-closed by design
        logger.warning("[content-gate] semantic inspection failed: %s", exc)
        return GateDecision(
            allowed=False,
            reason_code="content_inspection_unavailable",
            detail=f"semantic judge 调用失败：{exc}",
            judge_status="ERROR",
        )


async def semantic_review(content: str, *, context: dict[str, str] | None = None) -> GateDecision:
    """Judge-only review of produced content (structural layer already ran on
    the input side). Fail-closed: no judge, or any error, blocks. Used by the
    external-agent runtime supervisor when a run is halted/truncated."""
    resolved_context = dict(context or {})
    resolved_context.setdefault("run_id", "supervision-review")
    try:
        return await _inspect_semantic(str(content or ""), context=resolved_context)
    except Exception as exc:  # noqa: BLE001 — fail-closed by design
        logger.warning("[content-gate] semantic review failed: %s", exc)
        return GateDecision(
            allowed=False,
            reason_code="content_inspection_unavailable",
            detail=f"semantic judge 调用失败：{exc}",
            judge_status="ERROR",
        )


__all__ = [
    "GateDecision",
    "inspect_materialisation_content",
    "semantic_review",
]
