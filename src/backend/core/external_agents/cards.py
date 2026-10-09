"""A2A-standard AgentCards for the external agents.

Each external agent carries a maintainable card in the A2A (Agent2Agent)
AgentCard JSON format — camelCase spec fields (`protocolVersion`, `capabilities`,
`skills`, `defaultInputModes`, ...) — so the platform's agents speak the shape
external A2A tooling expects. Defaults are derived from the catalog; operators
edit and store overrides per agent. Unknown fields on save are preserved
(forward compatibility with newer spec versions); platform extensions live
under a non-spec `extensions.sumc` block that survives edits.

Storage: ``system_configs`` key ``external_agent_card.<agent_id>``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from core.db.models import SystemConfig

logger = logging.getLogger(__name__)

KEY_PREFIX = "external_agent_card."
GROUP_KEY = "external_agents"
A2A_PROTOCOL_VERSION = "0.3.0"
CARD_MEDIA_TYPES = ["text/plain", "application/json"]


def default_card(card: Any, *, model_name: str = "") -> dict[str, Any]:
    """Spec-shaped AgentCard derived from an external agent catalog entry."""
    return {
        "protocolVersion": A2A_PROTOCOL_VERSION,
        "name": card.name,
        "description": card.description,
        "url": f"/v1/external-agents/{card.agent_id}/run",
        "preferredTransport": "JSONRPCHttp",
        "version": "1.0.0",
        "provider": {
            "organization": card.project,
            "url": f"https://github.com/{card.project}",
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": list(CARD_MEDIA_TYPES),
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": f"{card.agent_id}-general-assistance",
                "name": card.name,
                "description": card.description,
                "tags": [card.framework, "general-assistance"],
                "examples": [],
            }
        ],
        "extensions": {
            "sumc": {
                "agent_id": card.agent_id,
                "framework": card.framework,
                "runtime_model": model_name or None,
                "model_configurable": True,
                "chat_selectable": True,
            }
        },
    }


def validate_card(payload: Any) -> dict[str, Any]:
    """Minimal spec validation; unknown keys survive untouched."""
    if not isinstance(payload, dict):
        raise ValueError("AgentCard 必须是 JSON 对象")
    for required in ("name", "description"):
        value = payload.get(required)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AgentCard 缺少必填字段: {required}")
    skills = payload.get("skills")
    if skills is not None and not isinstance(skills, list):
        raise ValueError("AgentCard.skills 必须是数组")
    input_modes = payload.get("defaultInputModes")
    if input_modes is not None and not (
        isinstance(input_modes, list)
        and all(isinstance(mode, str) for mode in input_modes)
    ):
        raise ValueError("AgentCard.defaultInputModes 必须是字符串数组")
    output_modes = payload.get("defaultOutputModes")
    if output_modes is not None and not (
        isinstance(output_modes, list)
        and all(isinstance(mode, str) for mode in output_modes)
    ):
        raise ValueError("AgentCard.defaultOutputModes 必须是字符串数组")
    return payload


def _key(agent_id: str) -> str:
    return f"{KEY_PREFIX}{agent_id}"


def get_card(db: Session, agent_id: str, catalog_card: Any) -> dict[str, Any]:
    """The stored override, else the spec-shaped default."""
    row = db.get(SystemConfig, _key(agent_id))
    if row is not None and (row.config_value or "").strip():
        try:
            stored = json.loads(row.config_value)
            if isinstance(stored, dict):
                return stored
        except json.JSONDecodeError:
            logger.warning("[external-agents] unreadable card override for %s", agent_id)
    return default_card(catalog_card)


def save_card(db: Session, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    validated = validate_card(payload)
    key = _key(agent_id)
    row = db.get(SystemConfig, key)
    body = json.dumps(validated, ensure_ascii=False)
    if row is None:
        row = SystemConfig(
            config_key=key,
            display_name=f"A2A AgentCard: {agent_id}",
            description="外部智能体的 A2A 标准 AgentCard（可单独维护）",
            group_key=GROUP_KEY,
            is_secret=False,
        )
        db.add(row)
    row.config_value = body
    db.commit()
    return validated


__all__ = [
    "A2A_PROTOCOL_VERSION",
    "default_card",
    "get_card",
    "save_card",
    "validate_card",
]
