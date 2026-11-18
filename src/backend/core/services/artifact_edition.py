"""Community artifact lists are always personal."""

from typing import Any, Optional

# Capability flag that marks a platform admin for artifact purposes (same flag
# the admin console gate uses; ``super_admin`` role implies it implicitly).
ARTIFACT_ADMIN_FLAG = "can_content_manage"


def artifact_list_scope() -> str:
    return "personal"


def extend_artifact_item(artifact, item: dict) -> dict:
    return item


def personal_artifact_predicates(artifact_model: Any) -> list[Any]:
    return []


def is_personal_artifact(artifact: Any) -> bool:
    return True


def personal_artifact_create_fields() -> dict[str, None]:
    return {}


def artifact_scope_fields(scope: Any) -> dict[str, Optional[str]]:
    return {
        "user_folder_id": (
            scope.root_folder_id or None if scope is not None and scope.is_personal else None
        )
    }


def artifact_scope_folder_id(scope: Any, artifact: Any) -> Optional[str]:
    return artifact.user_folder_id


def can_access_artifact(db: Any, user_id: str, artifact: Any) -> bool:
    return str(artifact.user_id) == str(user_id)


def artifact_access_metadata(artifact: Any) -> dict[str, Optional[str]]:
    return {"owner_id": artifact.user_id}


def _is_platform_admin(db: Any, user_id: str) -> bool:
    if db is None or not user_id:
        return False
    try:
        from core.auth.capabilities import user_has_capability

        return bool(user_has_capability(db, str(user_id), ARTIFACT_ADMIN_FLAG))
    except Exception:  # noqa: BLE001 — fail closed, never fail open
        return False


def can_access_artifact_metadata(db: Any, user_id: str, metadata: dict[str, Any]) -> bool:
    """Creator-attributed access for metadata-shaped artifacts (M14 phase 2).

    An owned item is readable by its owner only. An *ownerless* item (legacy
    store-index entries and tool products with no recorded creator) used to be
    readable by any authenticated caller — that hole closes here: without a
    creator to compare against, only a platform admin may read. Any failure
    resolving the admin capability degrades to deny (fail closed).
    """
    owner_id = metadata.get("owner_id") or metadata.get("user_id")
    if owner_id is not None:
        return str(owner_id) == str(user_id)
    return _is_platform_admin(db, user_id)


__all__ = [
    "artifact_access_metadata",
    "artifact_list_scope",
    "artifact_scope_fields",
    "artifact_scope_folder_id",
    "can_access_artifact",
    "can_access_artifact_metadata",
    "extend_artifact_item",
    "is_personal_artifact",
    "personal_artifact_create_fields",
    "personal_artifact_predicates",
]
