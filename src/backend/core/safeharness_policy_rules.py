"""DB-backed custom injection rules for the SafeHarness policy center.

Additive layer on top of ``safeharness.policies.rules_loader``: operators
manage a rule set in system_configs (key ``safeharness_policy_rules``); when
enabled it is hot-applied to the running process via ``apply_runtime_rules``
(no restart). The builtin defaults and any file-based rules
(``SAFEHARNESS_INFORM_RULES_DIR``) always stay in effect — this layer can only
add deny-patterns, never remove protection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from safeharness.policies.rules_loader import (
    apply_runtime_chains,
    apply_runtime_rules,
    clear_runtime_chains,
    clear_runtime_rules,
    runtime_chains,
    runtime_rules,
    validate_chain_dicts,
    validate_rule_dicts,
)

_CONFIG_KEY = "safeharness_policy_rules"
_GROUP_KEY = "safeharness"
_MAX_HISTORY = 10


class PolicyRulesValidationError(ValueError):
    """Raised with a list of human-readable validation errors."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors[:5]) + (" ..." if len(errors) > 5 else ""))


@dataclass
class PolicyRulesState:
    enabled: bool = False
    version: int = 0
    rules: list[dict[str, Any]] = field(default_factory=list)
    chains: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = ""
    updated_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "version": self.version,
            "rules": self.rules,
            "chains": self.chains,
            "history": self.history,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
        }


def load_policy_rules(db: Any) -> PolicyRulesState:
    """Stored JSON config → state; bad/absent rows fall back to defaults."""
    from core.db.models import SystemConfig

    try:
        row = db.get(SystemConfig, _CONFIG_KEY)
        if row is None or not (row.config_value or "").strip():
            return PolicyRulesState()
        raw = json.loads(row.config_value)
        if not isinstance(raw, dict):
            return PolicyRulesState()
        return PolicyRulesState(
            enabled=bool(raw.get("enabled", False)),
            version=max(0, int(raw.get("version", 0))),
            rules=list(raw.get("rules") or []),
            chains=list(raw.get("chains") or []),
            history=list(raw.get("history") or []),
            updated_at=str(raw.get("updated_at") or ""),
            updated_by=str(raw.get("updated_by") or ""),
        )
    except Exception:  # noqa: BLE001 — a broken config row must never open the gate
        return PolicyRulesState()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _persist(db: Any, state: PolicyRulesState) -> None:
    from core.db.models import SystemConfig

    row = db.get(SystemConfig, _CONFIG_KEY)
    body = json.dumps(state.to_dict(), ensure_ascii=False)
    if row is None:
        row = SystemConfig(
            config_key=_CONFIG_KEY,
            display_name="SafeHarness 自定义注入规则",
            description="补充式规则包：叠加在内置防线之上，仅支持 deny；启用后对本进程热生效",
            group_key=_GROUP_KEY,
            is_secret=False,
        )
        db.add(row)
    row.config_value = body
    db.commit()


def _apply_to_process(state: PolicyRulesState) -> None:
    if state.enabled and (state.rules or state.chains):
        apply_runtime_rules(state.rules)
        apply_runtime_chains(state.chains)
    else:
        clear_runtime_rules()
        clear_runtime_chains()


def resync_runtime_rules(db: Any) -> PolicyRulesState:
    """Re-apply the stored state to this process (used after a restart)."""
    state = load_policy_rules(db)
    _apply_to_process(state)
    return state


def apply_policy_rules(
    db: Any,
    *,
    rules: list[dict[str, Any]],
    enabled: bool,
    actor: str,
    chains: list[dict[str, Any]] | None = None,
) -> PolicyRulesState:
    """Validate → hot-apply → persist as a new version (old head → history)."""
    chain_list = list(chains or [])
    errors = validate_rule_dicts(rules) + validate_chain_dicts(chain_list)
    if errors:
        raise PolicyRulesValidationError(errors)
    state = load_policy_rules(db)
    state.version += 1
    state.rules = list(rules)
    state.chains = chain_list
    state.enabled = bool(enabled)
    state.updated_at = _now()
    state.updated_by = actor
    state.history.append(
        {
            "version": state.version,
            "rules": list(rules),
            "chains": chain_list,
            "enabled": state.enabled,
            "saved_at": state.updated_at,
            "saved_by": actor,
        }
    )
    state.history = state.history[-_MAX_HISTORY:]
    _apply_to_process(state)
    _persist(db, state)
    return state


def rollback_policy_rules(db: Any, *, version: int, actor: str) -> PolicyRulesState:
    """Re-apply a historical version's rules as a new head version."""
    state = load_policy_rules(db)
    entry = next(
        (item for item in state.history if int(item.get("version", -1)) == version),
        None,
    )
    if entry is None:
        raise KeyError(f"version {version} not in policy-rules history")
    return apply_policy_rules(
        db,
        rules=list(entry.get("rules") or []),
        enabled=bool(entry.get("enabled", False)),
        actor=actor,
        chains=list(entry.get("chains") or []),
    )


def effective_summary() -> dict[str, Any]:
    """What the process is actually enforcing right now (for the UI)."""
    return {
        "runtime_rule_count": len(runtime_rules()),
        "runtime_rules": [rule.name for rule in runtime_rules()],
        "runtime_chain_count": len(runtime_chains()),
        "runtime_chains": [chain.name for chain in runtime_chains()],
    }


__all__ = [
    "PolicyRulesState",
    "PolicyRulesValidationError",
    "apply_policy_rules",
    "effective_summary",
    "load_policy_rules",
    "resync_runtime_rules",
    "rollback_policy_rules",
]
