"""Runtime supervision for external agents — LLM-free filter, graded intervention.

Mechanism borrowed from LINs-lab/SupervisorAgent (ICLR 2026): a zero-cost
structural filter runs on every step/turn, and only risky runs escalate. The
LLM leg reuses the platform's semantic judge — no new model call surface, and
unlike that project we never absorb their smolagents fork.

Checks (in firing order):
1. ``max_steps``          — step/turn cap (framework-native caps are set from
                            this config too; the filter is the backstop).
2. ``loop_trigger``       — N consecutive near-identical observations → loop.
3. ``max_output_chars``   — single observation/output length.
4. ``max_total_chars``    — cumulative output budget.

Verdicts:
- ``allow``     — no intervention.
- ``truncate``  — the final reply exceeds a budget and is replaced by a
                  structured notice; escalation (semantic review of what was
                  produced) runs before the notice is finalised.
- ``halt``      — a mid-run intervention (loop / cap reached): the run stops
                  and the same escalation applies.

Escalation result is recorded by the registry into the audit ledger; the
review-workbench integration is deliberately out of scope for now (audited,
not queued).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_CONFIG_KEY = "external_agent_supervision"
_GROUP_KEY = "external_agents"

_TRIGGERS = ("max_steps", "loop", "output_length", "total_budget")


@dataclass(frozen=True)
class SupervisionConfig:
    max_steps: int = 8
    loop_trigger: int = 3
    max_output_chars: int = 8000
    max_total_chars: int = 24000

    def to_dict(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "loop_trigger": self.loop_trigger,
            "max_output_chars": self.max_output_chars,
            "max_total_chars": self.max_total_chars,
        }


DEFAULT_SUPERVISION = SupervisionConfig()


def load_supervision_config(db: Any) -> SupervisionConfig:
    """Stored JSON config → SupervisionConfig; bad/absent rows fall back to
    defaults (a broken config must never disable supervision)."""
    from core.db.models import SystemConfig

    try:
        row = db.get(SystemConfig, _CONFIG_KEY)
        if row is None or not (row.config_value or "").strip():
            return DEFAULT_SUPERVISION
        raw = json.loads(row.config_value)
        if not isinstance(raw, dict):
            return DEFAULT_SUPERVISION
        return SupervisionConfig(
            max_steps=max(1, int(raw.get("max_steps", DEFAULT_SUPERVISION.max_steps))),
            loop_trigger=max(2, int(raw.get("loop_trigger", DEFAULT_SUPERVISION.loop_trigger))),
            max_output_chars=max(200, int(raw.get("max_output_chars", DEFAULT_SUPERVISION.max_output_chars))),
            max_total_chars=max(1000, int(raw.get("max_total_chars", DEFAULT_SUPERVISION.max_total_chars))),
        )
    except Exception:  # noqa: BLE001 — fail closed to defaults, never open
        return DEFAULT_SUPERVISION


def save_supervision_config(db: Any, config: SupervisionConfig) -> None:
    from core.db.models import SystemConfig

    row = db.get(SystemConfig, _CONFIG_KEY)
    body = json.dumps(config.to_dict(), ensure_ascii=False)
    if row is None:
        row = SystemConfig(
            config_key=_CONFIG_KEY,
            display_name="外部智能体运行监督阈值",
            description="步数/循环/输出长度/累计预算；触发后截断并升级复查",
            group_key=_GROUP_KEY,
            is_secret=False,
        )
        db.add(row)
    row.config_value = body
    db.commit()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()[:160]


@dataclass
class SupervisionState:
    """Per-run supervisor: feed every observation/output through ``check``."""

    config: SupervisionConfig
    steps: int = 0
    total_chars: int = 0
    recent: list[str] = field(default_factory=list)
    interventions: list[dict[str, Any]] = field(default_factory=list)

    def check(self, *, output_text: str, final: bool = False) -> dict[str, Any] | None:
        """Record one step/output. Returns an intervention dict when the run
        must stop (or be truncated on the final output), else None."""
        self.steps += 1
        text = output_text or ""
        self.total_chars += len(text)

        def _intervene(trigger: str, detail: str) -> dict[str, Any]:
            intervention = {
                "trigger": trigger,
                "detail": detail,
                "step": self.steps,
                "truncated_chars": len(text),
            }
            self.interventions.append(intervention)
            return intervention

        if self.steps > self.config.max_steps:
            return _intervene("max_steps", f"步数超过上限 {self.config.max_steps}")
        if self.config.max_total_chars and self.total_chars > self.config.max_total_chars:
            return _intervene("total_budget", f"累计输出超预算 {self.config.max_total_chars}")
        if final and self.config.max_output_chars and len(text) > self.config.max_output_chars:
            return _intervene("output_length", f"单次输出超限 {self.config.max_output_chars}")

        normalised = _normalise(text)
        if normalised:
            self.recent.append(normalised)
            trigger = self.config.loop_trigger
            if len(self.recent) >= trigger and len(set(self.recent[-trigger:])) == 1:
                return _intervene("loop", f"连续 {trigger} 次近似相同输出")
        return None


def supervision_notice(intervention: dict[str, Any], *, reviewed: str = "not_reviewed") -> str:
    """The structured reply a truncated/halted run degrades to."""
    return (
        f"运行监督：该次执行已在第 {intervention.get('step')} 步终止"
        f"（原因：{intervention.get('trigger')} — {intervention.get('detail')}）。"
        f"输出安全复查：{reviewed}。可调整任务描述后重试。"
    )


__all__ = [
    "DEFAULT_SUPERVISION",
    "SupervisionConfig",
    "SupervisionState",
    "load_supervision_config",
    "save_supervision_config",
    "supervision_notice",
]
