"""Operator-registered custom external-agent cards — data, not code.

Registration records live in ``system_configs`` under keys
``external_agent_custom.<agent_id>`` (group ``external_agents``) and merge
with the builtin ``CATALOG`` at read time. Every reader hits the DB fresh
(same convention as the other external-agent config rows), so a register /
update / delete takes effect on the next request without a restart.

``framework`` is whitelist-enforced against ``RUNNERS`` at write time and
re-verified at read time: a record whose framework is no longer known is
excluded from the effective catalog, never executed. The runners themselves
stay code-side — a custom card can never introduce one.

The A2A card row (``external_agent_card.<agent_id>``) keeps its existing
meaning (an operator-editable card face); custom agents simply do not pre-write
one, and ``cards.get_card`` falls back to ``default_card`` for them.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from core.db.models import SystemConfig

logger = logging.getLogger(__name__)

KEY_PREFIX = "external_agent_custom."
GROUP_KEY = "external_agents"
AGENT_ID_RE = re.compile(r"^[a-z][a-z0-9-]{2,62}$")


class CustomAgentError(ValueError):
    """Registration validation failure (the message is user-facing)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(agent_id: str) -> str:
    return KEY_PREFIX + agent_id


def validate_agent_id(agent_id: str) -> str:
    from core.external_agents.registry import CATALOG

    if not isinstance(agent_id, str) or not AGENT_ID_RE.fullmatch(agent_id):
        raise CustomAgentError(
            "agent_id 只能是小写字母开头的 3-63 位小写字母/数字/连字符"
        )
    if agent_id in CATALOG:
        raise CustomAgentError(f"agent_id 与内置智能体冲突: {agent_id}")
    return agent_id


def _validate_fields(*, name: str, framework: str, project: str, description: str) -> None:
    from core.external_agents.runners import RUNNERS

    if not isinstance(name, str) or not (1 <= len(name) <= 64):
        raise CustomAgentError("name 需为 1-64 字")
    if framework not in RUNNERS:
        raise CustomAgentError(
            f"framework 必须是受支持框架之一: {', '.join(sorted(RUNNERS))}"
        )
    if len(project) > 200:
        raise CustomAgentError("project 最长 200 字")
    if not isinstance(description, str) or not (1 <= len(description) <= 500):
        raise CustomAgentError("description 需为 1-500 字")


def _record_from_row(agent_id: str, raw: str | None) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[external-agents] broken custom record ignored: %s", agent_id)
        return None
    if not isinstance(payload, dict):
        return None
    record = dict(payload)
    record["agent_id"] = agent_id
    record.setdefault("enabled", True)
    return record


def list_custom(db) -> dict[str, dict[str, Any]]:
    """All custom records keyed by agent_id (broken rows are skipped).

    Fail-closed to an empty catalog on any read failure — a degraded DB must
    degrade to "builtin only", never break listing or dispatch.
    """
    try:
        rows = (
            db.query(SystemConfig)
            .filter(SystemConfig.config_key.like(KEY_PREFIX + "%"))
            .all()
        )
    except Exception as exc:  # noqa: BLE001 — same convention as memory/supervision reads
        logger.warning("[external-agents] custom catalog read failed: %s", type(exc).__name__)
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        agent_id = row.config_key[len(KEY_PREFIX):]
        record = _record_from_row(agent_id, row.config_value)
        if record is not None:
            out[agent_id] = record
    return out


def get_custom(db, agent_id: str) -> dict[str, Any] | None:
    try:
        row = db.get(SystemConfig, _key(agent_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[external-agents] custom record read failed: %s", type(exc).__name__)
        return None
    return _record_from_row(agent_id, row.config_value if row else None)


def register_custom(
    db,
    *,
    agent_id: str,
    name: str,
    framework: str,
    project: str = "",
    description: str,
    actor: str = "",
) -> dict[str, Any]:
    validate_agent_id(agent_id)
    _validate_fields(name=name, framework=framework, project=project, description=description)
    if get_custom(db, agent_id) is not None:
        raise CustomAgentError(f"agent_id 已存在: {agent_id}")
    now = _now()
    record = {
        "agent_id": agent_id,
        "name": name,
        "framework": framework,
        "project": project,
        "description": description,
        "enabled": True,
        "created_by": actor,
        "created_at": now,
        "updated_by": actor,
        "updated_at": now,
    }
    _write(db, agent_id, record)
    return record


def update_custom(db, agent_id: str, patch: dict[str, Any], *, actor: str = "") -> dict[str, Any]:
    record = get_custom(db, agent_id)
    if record is None:
        raise CustomAgentError(f"自定义智能体不存在: {agent_id}")
    fields = {
        "name": record["name"],
        "framework": record["framework"],
        "project": record.get("project") or "",
        "description": record["description"],
    }
    for key in ("name", "framework", "project", "description"):
        if key in patch and patch[key] is not None:
            fields[key] = str(patch[key])
    _validate_fields(
        name=fields["name"],
        framework=fields["framework"],
        project=fields["project"],
        description=fields["description"],
    )
    record.update(fields)
    if "enabled" in patch and patch["enabled"] is not None:
        record["enabled"] = bool(patch["enabled"])
    record["updated_by"] = actor
    record["updated_at"] = _now()
    _write(db, agent_id, record)
    return record


def delete_custom(db, agent_id: str) -> None:
    """Remove the record plus its card / model-override residue (idempotent)."""
    from core.external_agents.registry import CATALOG

    if agent_id in CATALOG:
        raise CustomAgentError("内置智能体不可删除")
    row = db.get(SystemConfig, _key(agent_id))
    if row is None:
        raise CustomAgentError(f"自定义智能体不存在: {agent_id}")
    db.delete(row)
    for residue in (
        f"external_agent_card.{agent_id}",
        f"external_agent_model.agent.{agent_id}",
    ):
        stale = db.get(SystemConfig, residue)
        if stale is not None:
            db.delete(stale)
    db.commit()


def _write(db, agent_id: str, record: dict[str, Any]) -> None:
    row = db.get(SystemConfig, _key(agent_id))
    if row is None:
        row = SystemConfig(
            config_key=_key(agent_id),
            display_name=f"自定义外部智能体: {agent_id}",
            description="外部智能体目录注册记录（数据，非代码）",
            group_key=GROUP_KEY,
            is_secret=False,
        )
        db.add(row)
    row.config_value = json.dumps(record, ensure_ascii=False)
    row.updated_by = str(record.get("updated_by") or "")[:64]
    db.commit()


def card_for_record(record: dict[str, Any]):
    """The catalog card for one custom record, or None when not runnable."""
    from core.external_agents.registry import ExternalAgentCard
    from core.external_agents.runners import RUNNERS

    framework = str(record.get("framework") or "")
    runner = RUNNERS.get(framework)
    if runner is None:
        return None
    return ExternalAgentCard(
        agent_id=str(record.get("agent_id") or ""),
        name=str(record.get("name") or ""),
        framework=framework,
        project=str(record.get("project") or ""),
        description=str(record.get("description") or ""),
        runner=runner,
    )


def merged_catalog(db) -> dict[str, Any]:
    """builtin CATALOG ∪ enabled custom cards (unknown-framework rows dropped)."""
    from core.external_agents.registry import CATALOG

    merged: dict[str, Any] = dict(CATALOG)
    for agent_id, record in list_custom(db).items():
        if not record.get("enabled", True):
            continue
        card = card_for_record(record)
        if card is not None:
            merged[agent_id] = card
    return merged


def lookup_agent(db, agent_id: str):
    """One card from the merged catalog, or None."""
    from core.external_agents.registry import CATALOG

    if agent_id in CATALOG:
        return CATALOG[agent_id]
    record = get_custom(db, agent_id)
    if record is None or not record.get("enabled", True):
        return None
    return card_for_record(record)


__all__ = [
    "AGENT_ID_RE",
    "CustomAgentError",
    "KEY_PREFIX",
    "card_for_record",
    "delete_custom",
    "get_custom",
    "list_custom",
    "lookup_agent",
    "merged_catalog",
    "register_custom",
    "update_custom",
]
