"""Memory configuration for external framework agents.

Two memory layers, one config row (``system_configs`` key
``external_agent_memory``):

* **session window** — the last N turns of this chat, injected per framework
  (native multi-turn input for openai-agents, ListMemory for autogen, a
  transcript prefix for smolagents);
* **cross-session long-term memory** — mem0 retrieval (read) and conversation
  save (write) keyed by the platform user id, guarded by a character budget
  (a practical stand-in for a token budget at the injection point).

Bad or partial config rows fall back to the defaults and never disable the
session window entirely; the long-term layer is additionally gated by the
platform's ``settings.memory.enabled`` (MEM0_ENABLED), so absence of the mem0
deployment degrades to window-only memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_CONFIG_KEY = "external_agent_memory"


@dataclass(frozen=True)
class MemoryConfig:
    history_window: int = 12
    long_term_enabled: bool = True
    long_term_max_chars: int = 2000
    long_term_limit: int = 5


_DEFAULTS = MemoryConfig()
_WINDOW_RANGE = (0, 50)
_CHARS_RANGE = (0, 16000)
_LIMIT_RANGE = (1, 20)


def _clamp_int(value: Any, fallback: int, bounds: tuple[int, int]) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    low, high = bounds
    return max(low, min(high, number))


def parse_memory_config(raw: Any) -> MemoryConfig:
    """Validate a stored row; any bad field falls back to its default."""
    if not isinstance(raw, dict):
        return _DEFAULTS
    enabled = raw.get("long_term_enabled", _DEFAULTS.long_term_enabled)
    if not isinstance(enabled, bool):
        enabled = _DEFAULTS.long_term_enabled
    return MemoryConfig(
        history_window=_clamp_int(
            raw.get("history_window"), _DEFAULTS.history_window, _WINDOW_RANGE
        ),
        long_term_enabled=enabled,
        long_term_max_chars=_clamp_int(
            raw.get("long_term_max_chars"),
            _DEFAULTS.long_term_max_chars,
            _CHARS_RANGE,
        ),
        long_term_limit=_clamp_int(
            raw.get("long_term_limit"), _DEFAULTS.long_term_limit, _LIMIT_RANGE
        ),
    )


def load_memory_config(db: Any) -> MemoryConfig:
    """Read the config row; missing row or unreadable storage yields defaults."""
    try:
        from core.db.models import SystemConfig

        row = db.get(SystemConfig, _CONFIG_KEY)
        if row is None or not (row.config_value or "").strip():
            return _DEFAULTS
        import json

        raw = json.loads(row.config_value)
        return parse_memory_config(raw)
    except Exception:  # noqa: BLE001 — config is best-effort, defaults are safe
        return _DEFAULTS


def save_memory_config(db: Any, patch: dict[str, Any]) -> MemoryConfig:
    """Merge a patch into the stored row (partial updates allowed)."""
    import json

    from core.db.models import SystemConfig

    current = load_memory_config(db)
    merged = parse_memory_config({**asdict(current), **(patch or {})})
    row = db.get(SystemConfig, _CONFIG_KEY)
    payload = json.dumps(asdict(merged), ensure_ascii=False)
    if row is None:
        row = SystemConfig(
            config_key=_CONFIG_KEY,
            config_value=payload,
            display_name="外部智能体记忆配置",
            description="会话窗口条数/跨会话长期记忆开关与字符预算",
            group_key="external_agents",
            is_secret=False,
        )
        db.add(row)
    else:
        row.config_value = payload
    db.commit()
    return merged


__all__ = [
    "MemoryConfig",
    "load_memory_config",
    "parse_memory_config",
    "save_memory_config",
]
