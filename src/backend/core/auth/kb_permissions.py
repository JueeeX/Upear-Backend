"""Tenant-scoped knowledge-base authorization."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from sqlalchemy.orm import Session

KBLevel = Literal["none", "view", "edit", "admin"]
_RANK = {"none": 0, "view": 1, "edit": 2, "admin": 3}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


def has_kb_permission(current: str, required: str) -> bool:
    return _RANK.get(current, 0) >= _RANK.get(required, 0)


def is_shared_visibility(visibility: str) -> bool:
    return visibility != "private"


def level_to_caps(level: str) -> Dict[str, bool]:
    return {
        "can_view": has_kb_permission(level, "view"),
        "can_edit": has_kb_permission(level, "edit"),
        "can_admin": has_kb_permission(level, "admin"),
    }


def resolve_user_kb_context(db: Session, user_id: str) -> Optional[Tuple[str, str]]:
    """Return explicit tenant and role metadata; never synthesize an authority scope."""
    from core.db.models import UserShadow

    if not user_id:
        return None
    shadow = db.get(UserShadow, str(user_id))
    metadata = shadow.extra_data if shadow and isinstance(shadow.extra_data, dict) else {}
    tenant_id = metadata.get("tenant_id")
    role = metadata.get("role")
    if not isinstance(tenant_id, str) or _IDENTIFIER.fullmatch(tenant_id.strip()) is None:
        return None
    if not isinstance(role, str) or _IDENTIFIER.fullmatch(role.strip()) is None:
        return None
    return tenant_id.strip(), role.strip()


def resolve_space_kb_tenant_id(db: Session, space: Any) -> Optional[str]:
    """Resolve a space tenant from authoritative metadata or its legacy owner identity."""
    metadata = space.extra_data if isinstance(getattr(space, "extra_data", None), dict) else {}
    tenant_id = metadata.get("tenant_id")
    binding = metadata.get("tenant_binding")
    if (
        isinstance(binding, dict)
        and binding.get("version") == 1
        and binding.get("source") == "authoritative_user_context"
        and isinstance(tenant_id, str)
        and _IDENTIFIER.fullmatch(tenant_id.strip()) is not None
    ):
        return tenant_id.strip()
    owner_context = resolve_user_kb_context(db, str(getattr(space, "user_id", "")))
    return owner_context[0] if owner_context else None


def get_accessible_local_kb_levels(db: Session, user_id: str) -> Dict[str, KBLevel]:
    from core.db.models import KBSpace

    context = resolve_user_kb_context(db, user_id)
    if context is None:
        return {}
    tenant_id, role = context
    is_admin = role == "super_admin"
    rows = (
        db.query(KBSpace)
        .filter(KBSpace.deleted_at.is_(None))
        .all()
    )
    return {
        space.kb_id: "admin" if is_admin or space.user_id == user_id else "view"
        for space in rows
        if resolve_space_kb_tenant_id(db, space) == tenant_id
        and (is_admin or space.user_id == user_id or is_shared_visibility(space.visibility))
    }


def get_accessible_local_kb_ids(db: Session, user_id: str) -> Set[str]:
    return set(get_accessible_local_kb_levels(db, user_id))


def get_dataset_levels(db: Session, user_id: str, dataset_ids: List[str]) -> Dict:
    return {}


def filter_accessible_kb_ids(db: Session, user_id: str, kb_ids: List[str]) -> List[str]:
    allowed = get_accessible_local_kb_ids(db, user_id)
    return [kb_id for kb_id in kb_ids if kb_id in allowed]


def resolve_local_kb_level(db: Session, user_id: str, kb_id: str) -> KBLevel:
    return get_accessible_local_kb_levels(db, user_id).get(kb_id, "none")


__all__ = [
    "KBLevel",
    "filter_accessible_kb_ids",
    "get_accessible_local_kb_ids",
    "get_accessible_local_kb_levels",
    "get_dataset_levels",
    "has_kb_permission",
    "is_shared_visibility",
    "level_to_caps",
    "resolve_local_kb_level",
    "resolve_space_kb_tenant_id",
    "resolve_user_kb_context",
]
