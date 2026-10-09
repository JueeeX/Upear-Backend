"""Per-agent and shared model settings for external agents.

Storage: ``system_configs`` rows (group ``external_agents``). Resolution order
for one agent run: agent-specific override → shared default → the platform
``main_agent`` provider. Provider ids are validated against the configured
model providers at write time, so a stale id can never sit in the setting.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from core.db.models import SystemConfig

logger = logging.getLogger(__name__)

DEFAULT_KEY = "external_agent_model.default"
AGENT_KEY_PREFIX = "external_agent_model.agent."
GROUP_KEY = "external_agents"


def _load(db: Session, key: str) -> Optional[str]:
    row = db.get(SystemConfig, key)
    if row is None or not (row.config_value or "").strip():
        return None
    try:
        value = json.loads(row.config_value)
    except json.JSONDecodeError:
        return None
    provider_id = value.get("provider_id") if isinstance(value, dict) else None
    return str(provider_id) if provider_id else None


def _save(db: Session, key: str, provider_id: Optional[str], display: str, description: str) -> None:
    row = db.get(SystemConfig, key)
    if provider_id is None:
        if row is not None:
            db.delete(row)
        return
    payload = json.dumps({"provider_id": provider_id}, ensure_ascii=False)
    if row is None:
        row = SystemConfig(
            config_key=key,
            display_name=display,
            description=description,
            group_key=GROUP_KEY,
            is_secret=False,
        )
        db.add(row)
    row.config_value = payload


def _validate_provider(db: Session, provider_id: str) -> None:
    from core.services.model_config import ModelConfigService

    resolved = ModelConfigService.get_instance().resolve_provider(provider_id)
    if resolved is None:
        raise ValueError(f"未知模型供应商: {provider_id}")


def get_model_settings(db: Session) -> dict[str, Any]:
    default = _load(db, DEFAULT_KEY)
    overrides: dict[str, str] = {}
    rows = (
        db.query(SystemConfig)
        .filter(SystemConfig.config_key.like(f"{AGENT_KEY_PREFIX}%"))
        .all()
    )
    for row in rows:
        agent_id = row.config_key[len(AGENT_KEY_PREFIX):]
        provider_id = _load(db, row.config_key)
        if agent_id and provider_id:
            overrides[agent_id] = provider_id
    return {"default_provider_id": default, "agents": overrides}


def set_default_model(db: Session, provider_id: Optional[str]) -> None:
    if provider_id is not None:
        _validate_provider(db, provider_id)
    _save(
        db,
        DEFAULT_KEY,
        provider_id,
        "外部智能体统一模型",
        "所有外部智能体默认使用的模型供应商；留空表示跟随平台主模型。",
    )
    db.commit()


def set_agent_model(db: Session, agent_id: str, provider_id: Optional[str]) -> None:
    if provider_id is not None:
        _validate_provider(db, provider_id)
    _save(
        db,
        f"{AGENT_KEY_PREFIX}{agent_id}",
        provider_id,
        f"外部智能体模型覆盖: {agent_id}",
        "该外部智能体单独使用的模型供应商；留空表示跟随统一模型。",
    )
    db.commit()


def resolve_provider_chain(db: Session, agent_id: str) -> tuple[Optional[str], str]:
    """(provider_id or None, source) — None means "fall back to main_agent"."""
    settings = get_model_settings(db)
    override = settings["agents"].get(agent_id)
    if override:
        return override, "agent_override"
    if settings["default_provider_id"]:
        return settings["default_provider_id"], "shared_default"
    return None, "main_agent"


__all__ = [
    "DEFAULT_KEY",
    "get_model_settings",
    "set_default_model",
    "set_agent_model",
    "resolve_provider_chain",
]
