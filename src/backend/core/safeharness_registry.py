"""Implementation registry for the SafeHarness judge and audit sink (P1.2).

Controlled pluginization, second step (P1.1 was the middleware registry):
config selects *implementations by name*; the registry never introduces code.
The failure direction is the opposite of the middleware registry's fail-open:
an unknown name here falls back to the **default implementation** — a missing
judge would disable the semantic layer, a missing audit sink would drop the
ledger, and neither is acceptable.

Selection lives in the ``impl_registry`` system_configs key, fresh-read:

    {"judge_provider": "openai_compatible" | "paranoid",
     "audit_sink": "auto" | "sqlite" | "memory"}

Judge entries:
- ``openai_compatible`` (default): no-op — every construction site keeps its
  own resolution order and caller stack, bit-identical.
- ``paranoid``: the semantic layer conservatively withholds approval — every
  judge request returns UNCERTAIN (judge_status "ok", zero model calls). For
  deployments with no model budget or during provider outages: the semantic
  gate fails closed while the structural layer keeps working.

Audit sink entries (consumed by core.safeharness_audit.build_audit_sink):
- ``auto`` (default): the historic resolution — explicit
  ``SAFEHARNESS_AUDIT_DB`` or the sqlite-sibling path, in-memory when empty.
- ``sqlite``: force the SQLite resolution (still falls back to memory with a
  warning when no path resolves).
- ``memory``: force the in-memory sink (nothing persists across restarts).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

IMPL_REGISTRY_CONFIG_KEY = "impl_registry"

JUDGE_OPENAI_COMPATIBLE = "openai_compatible"
JUDGE_PARANOID = "paranoid"
JUDGE_DEFAULT = JUDGE_OPENAI_COMPATIBLE

AUDIT_AUTO = "auto"
AUDIT_SQLITE = "sqlite"
AUDIT_MEMORY = "memory"
AUDIT_DEFAULT = AUDIT_AUTO

JUDGE_IMPLEMENTATIONS = (
    {"name": JUDGE_OPENAI_COMPATIBLE, "default": True, "description": "模型语义 Judge（judge/main_agent 角色回落，各站点保留自己的候选序）"},
    {"name": JUDGE_PARANOID, "default": False, "description": "保守模式：语义层恒 UNCERTAIN（fail-closed 全拦，零模型调用）"},
)

AUDIT_SINK_IMPLEMENTATIONS = (
    {"name": AUDIT_AUTO, "default": True, "description": "自动：显式路径→SQLite，否则内存（现行行为逐位一致）"},
    {"name": AUDIT_SQLITE, "default": False, "description": "强制 SQLite 解析（无路径时回落内存并告警）"},
    {"name": AUDIT_MEMORY, "default": False, "description": "强制内存（不持久，重启即失）"},
)


def _known(kind: str) -> tuple:
    source = JUDGE_IMPLEMENTATIONS if kind == "judge" else AUDIT_SINK_IMPLEMENTATIONS
    return tuple(entry["name"] for entry in source)


def load_selection(db=None) -> dict:
    """The validated selection dict, fresh-read, unknown names → default.

    ``db`` is an injection seam for tests; production opens its own session
    (same fresh-read convention as tenant_quota / middleware_overrides).
    """
    selection = {"judge_provider": JUDGE_DEFAULT, "audit_sink": AUDIT_DEFAULT}
    try:
        if db is not None:
            row = db.get(_system_config_model(), IMPL_REGISTRY_CONFIG_KEY)
            raw = row.config_value if row else None
        else:
            from core.db.engine import SessionLocal

            with SessionLocal() as session:
                row = session.get(_system_config_model(), IMPL_REGISTRY_CONFIG_KEY)
            raw = row.config_value if row else None
        if not raw or not raw.strip():
            return selection
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return selection
        judge = str(payload.get("judge_provider") or "").strip()
        sink = str(payload.get("audit_sink") or "").strip()
        if judge and judge in _known("judge"):
            selection["judge_provider"] = judge
        elif judge:
            logger.warning("[impl-registry] unknown judge_provider %r — using default", judge)
        if sink and sink in _known("audit"):
            selection["audit_sink"] = sink
        elif sink:
            logger.warning("[impl-registry] unknown audit_sink %r — using default", sink)
        return selection
    except Exception as exc:  # noqa: BLE001 — fail safe to defaults
        logger.warning("[impl-registry] selection read failed: %s", type(exc).__name__)
        return selection


def selected_judge_implementation(db=None) -> str:
    return load_selection(db)["judge_provider"]


def selected_audit_sink_implementation(db=None) -> str:
    return load_selection(db)["audit_sink"]


def _system_config_model():
    from core.db.models import SystemConfig

    return SystemConfig


class ParanoidJudge:
    """The semantic layer's conservative mode: every request is UNCERTAIN.

    Zero model calls, fail-closed downstream (a non-executable judge decision
    blocks the semantic gate while the structural layer keeps working). The
    instance is a process-wide singleton by construction — it holds no state.
    """

    provider = "paranoid"
    model = "paranoid"
    last_status = "UNCERTAIN"

    async def judge(self, request: Mapping[str, Any] | Any) -> Any:
        from safeharness.ports.base import JudgeResult

        return JudgeResult(
            label="UNCERTAIN",
            confidence=0.0,
            reason="impl-registry paranoid mode: the semantic layer conservatively withholds approval",
            provider=self.provider,
            model=self.model,
            judge_status="ok",
            judge_calls=0,
            judge_result_class="unspecified",
        )


_PARANOID_SINGLETON: Optional[ParanoidJudge] = None


def paranoid_judge() -> ParanoidJudge:
    global _PARANOID_SINGLETON
    if _PARANOID_SINGLETON is None:
        _PARANOID_SINGLETON = ParanoidJudge()
    return _PARANOID_SINGLETON


__all__ = [
    "AUDIT_DEFAULT",
    "AUDIT_SINK_IMPLEMENTATIONS",
    "IMPL_REGISTRY_CONFIG_KEY",
    "JUDGE_DEFAULT",
    "JUDGE_IMPLEMENTATIONS",
    "ParanoidJudge",
    "load_selection",
    "paranoid_judge",
    "selected_audit_sink_implementation",
    "selected_judge_implementation",
]
