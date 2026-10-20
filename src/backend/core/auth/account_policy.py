"""Single-tenant account policy for the community edition."""

from core.infra.exceptions import AppException


class AccountCapacityExceeded(AppException):
    """Neutral account-admission exception; CE never raises it."""


def account_capacity_block_reason(db):
    """Admission check for new accounts (M14 quota framework, phase 1).

    The creation seam cannot know a not-yet-created user's tenant, so phase 1
    enforces the ``default`` tenant bucket only — shadows without an explicit
    ``tenant_id`` declaration (or declared as default). Capacity for declared
    tenants needs the tenant-assignment flow and lands with the phase-2
    column work. Fail-open: no quota configured means unlimited, which is the
    pre-M14 behaviour bit-identical.
    """
    try:
        from core.auth.tenancy import DEFAULT_TENANT, TENANT_METADATA_KEY, load_tenant_quota

        quota = load_tenant_quota(DEFAULT_TENANT)
        try:
            max_users = int(quota.get("max_users") or 0)
        except (TypeError, ValueError):
            max_users = 0
        if max_users <= 0:
            return None

        from core.db.models import UserShadow

        total = 0
        declared_other = 0
        for (extra,) in db.query(UserShadow.extra_data).all():
            total += 1
            metadata = extra if isinstance(extra, dict) else {}
            declared = str(metadata.get(TENANT_METADATA_KEY) or "").strip()
            if declared and declared != DEFAULT_TENANT:
                declared_other += 1
        used = total - declared_other
        if used < max_users:
            return None
        return f"租户账号数已达配额上限（{used}/{max_users}）"
    except Exception:  # noqa: BLE001 — admission must fail open, never break signup
        return None


def validate_registration_credential(db, credential: str):
    return True, None, None


def claim_registration_credential(db, credential: str, user_id: str):
    return True, None


def registration_credential_id(validated) -> None:
    return None


def add_invited_account_to_scope(db, validated, user_id: str) -> None:
    return None


def validate_account_scope(db, scope_id) -> bool:
    return scope_id is None


def add_account_to_scope(db, scope_id, user_id: str, role: str) -> None:
    return None


def list_account_scopes(db, user_id: str) -> list:
    return []


__all__ = [
    "AccountCapacityExceeded",
    "account_capacity_block_reason",
    "add_account_to_scope",
    "add_invited_account_to_scope",
    "claim_registration_credential",
    "list_account_scopes",
    "registration_credential_id",
    "validate_account_scope",
    "validate_registration_credential",
]
