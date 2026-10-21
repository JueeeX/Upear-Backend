"""Declarative middleware assembly registry (P1.1, controlled pluginization).

The agent factory used to hardcode the ``_middlewares`` list inline. The
registry stores *specs*, never instances — every middleware carries per-run
mutable state (injection flags, counters, fired markers), so each agent build
must construct its own fresh instances.

Two levels:

- ``security``: the SafeHarness tool-policy middleware and the ontology
  contract gate. The registry refuses to disable these — an override asking
  for it is ignored with a warning, never honored (特权安全核心不可插).
  Their relative order is pinned: both implement ``on_acting`` and the
  ontology gate is outermost of the two today.
- ``functional``: everything else. Operators may disable these by name via
  the ``middleware_overrides`` system_configs key:
  ``{"disabled": ["workspace_pin_hint", ...]}`` — fresh-read at every agent
  assembly (agents are rebuilt per request), so a change lands on the next
  message without a restart. Malformed input fails open (empty override set)
  with a warning.

Out of scope by design: operator-defined ordering (the on_acting gate
semantics depend on it) and operator-added classes (middleware classes are
code-side — a config entry can never introduce one; same split as the
external-agent catalog's "cards are data, runners are code").
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List

logger = logging.getLogger(__name__)

LEVEL_SECURITY = "security"
LEVEL_FUNCTIONAL = "functional"

OVERRIDES_CONFIG_KEY = "middleware_overrides"


@dataclass(frozen=True)
class AssemblyContext:
    """Per-run knobs the specs' gates and builders read."""

    batch_mode: bool = False
    chat_id: str | None = None
    intervention_rules: List[Any] = field(default_factory=list)
    ontology_runtime: Any = None


@dataclass(frozen=True)
class MiddlewareSpec:
    """One registry entry: how to build a fresh instance, and when."""

    name: str
    level: str
    build: Callable[[AssemblyContext], Any]
    enabled: Callable[[AssemblyContext], bool] = lambda ctx: True


def _build_registry() -> tuple:
    """The default registry, in the factory's historic assembly order.

    Order and construction arguments are load-bearing: with no overrides the
    assembled list must be identical to the pre-registry hardcoding, entry for
    entry (class, constructor kwargs, gates).
    """
    from core.llm.middlewares import (
        ActingToolCallIdMiddleware,
        DynamicModelMiddleware,
        FileContextMiddleware,
        FinishPinGuardMiddleware,
        GoalAnchorReminderMiddleware,
        IterBudgetReminderMiddleware,
        OntologyGateMiddleware,
        StallInterventionMiddleware,
        WorkspacePinHintMiddleware,
    )
    from core.llm.safeharness_mcp_guard import SafeHarnessToolPolicyMiddleware

    return (
        MiddlewareSpec(
            "dynamic_model",
            LEVEL_FUNCTIONAL,
            lambda ctx: DynamicModelMiddleware(),
        ),
        MiddlewareSpec(
            "file_context",
            LEVEL_FUNCTIONAL,
            lambda ctx: FileContextMiddleware(),
        ),
        MiddlewareSpec(
            "workspace_pin_hint",
            LEVEL_FUNCTIONAL,
            lambda ctx: WorkspacePinHintMiddleware(),
        ),
        MiddlewareSpec(
            "iter_budget_reminder",
            LEVEL_FUNCTIONAL,
            lambda ctx: IterBudgetReminderMiddleware(),
        ),
        MiddlewareSpec(
            "stall_intervention",
            LEVEL_FUNCTIONAL,
            lambda ctx: StallInterventionMiddleware(ctx.intervention_rules),
        ),
        MiddlewareSpec(
            "ontology_gate",
            LEVEL_SECURITY,
            lambda ctx: OntologyGateMiddleware(ctx.ontology_runtime),
        ),
        MiddlewareSpec(
            "acting_tool_call_id",
            LEVEL_FUNCTIONAL,
            lambda ctx: ActingToolCallIdMiddleware(),
        ),
        MiddlewareSpec(
            "goal_anchor",
            LEVEL_FUNCTIONAL,
            lambda ctx: GoalAnchorReminderMiddleware(chat_id=ctx.chat_id, batch_mode=False),
            enabled=lambda ctx: not ctx.batch_mode,
        ),
        MiddlewareSpec(
            "finish_pin",
            LEVEL_FUNCTIONAL,
            lambda ctx: FinishPinGuardMiddleware(batch_mode=ctx.batch_mode),
        ),
        MiddlewareSpec(
            "safeharness_tool_policy",
            LEVEL_SECURITY,
            lambda ctx: SafeHarnessToolPolicyMiddleware(),
            enabled=lambda ctx: _policy_guard_enabled(),
        ),
    )


def _policy_guard_enabled() -> bool:
    from core.llm.safeharness_mcp_guard import policy_guard_enabled

    return policy_guard_enabled()


DEFAULT_REGISTRY = _build_registry()
REGISTRY_BY_NAME = {spec.name: spec for spec in DEFAULT_REGISTRY}
FUNCTIONAL_NAMES = tuple(s.name for s in DEFAULT_REGISTRY if s.level == LEVEL_FUNCTIONAL)


def load_disabled_names(db=None) -> set:
    """Operator-disabled functional middleware names, fresh-read, fail-open."""
    try:
        if db is not None:
            row = db.get(_system_config_model(), OVERRIDES_CONFIG_KEY)
            return _parse_disabled(row.config_value if row else None)
        from core.db.engine import SessionLocal

        with SessionLocal() as session:
            row = session.get(_system_config_model(), OVERRIDES_CONFIG_KEY)
            return _parse_disabled(row.config_value if row else None)
    except Exception as exc:  # noqa: BLE001 — fail open, never break assembly
        logger.warning("[middleware-registry] override read failed: %s", type(exc).__name__)
        return set()


def _system_config_model():
    from core.db.models import SystemConfig

    return SystemConfig


def _parse_disabled(raw: str | None) -> set:
    if not raw or not raw.strip():
        return set()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[middleware-registry] malformed override JSON ignored")
        return set()
    if not isinstance(payload, dict):
        return set()
    requested = payload.get("disabled")
    if not isinstance(requested, list):
        return set()
    disabled: set = set()
    for name in requested:
        spec = REGISTRY_BY_NAME.get(str(name))
        if spec is None:
            logger.warning("[middleware-registry] unknown middleware name ignored: %s", name)
            continue
        if spec.level == LEVEL_SECURITY:
            logger.warning(
                "[middleware-registry] refusing to disable security middleware: %s", name
            )
            continue
        disabled.add(spec.name)
    return disabled


def assemble_middlewares(ctx: AssemblyContext, *, db=None) -> List[Any]:
    """Assemble one agent's middleware list, honouring operator overrides.

    With no overrides this is bit-identical to the pre-registry hardcoding.
    ``db`` is an injection seam for tests.
    """
    disabled = load_disabled_names(db)

    out: List[Any] = []
    for spec in DEFAULT_REGISTRY:
        if spec.name in disabled:
            continue
        if not spec.enabled(ctx):
            continue
        out.append(spec.build(ctx))
    return out


__all__ = [
    "AssemblyContext",
    "DEFAULT_REGISTRY",
    "FUNCTIONAL_NAMES",
    "LEVEL_FUNCTIONAL",
    "LEVEL_SECURITY",
    "MiddlewareSpec",
    "OVERRIDES_CONFIG_KEY",
    "assemble_middlewares",
    "load_disabled_names",
]
