"""Fail-closed SafeHarness read model for SUMC capabilities."""

from __future__ import annotations

from typing import Any, Mapping, NoReturn


def reject_uncontrolled_mutation() -> NoReturn:
    """Block legacy write routes that bypass SafeHarness approval."""
    from core.infra.exceptions import AccessDeniedError

    raise AccessDeniedError(
        message="SafeHarness controlled mutation required",
        reason="Use /v1/safeharness/mutations with re-authentication and one-shot approval.",
    )


def capability_security_metadata(
    *, kind: str, identifier: object, source_type: object
) -> dict[str, Any]:
    """Expose only authoritative provenance; unavailable security facts stay unknown."""
    source_id = "" if identifier is None else str(identifier).strip()
    source = "" if source_type is None else str(source_type).strip()
    result = {
        "schema_version": "1.0",
        "source": {
            "type": source or "unknown",
            "kind": str(kind).strip() or "unknown",
            "id": source_id or None,
        },
        "tool_descriptor": {"status": "unknown"},
        "adapter_coverage": {"status": "unknown"},
        "policy_package": {"status": "unknown"},
        "execution": {
            "status": "blocked",
            "reason_code": "safeharness_capability_not_mapped",
        },
    }
    from core.safeharness_capability_mapping import capability_mapping

    mapped = capability_mapping(result["source"]["id"], kind=result["source"]["kind"])
    if mapped is not None:
        result.update(mapped)
    return result


def with_capability_security(
    item: Mapping[str, Any], *, kind: str, identifier_key: str, source_type: object
) -> dict[str, Any]:
    result = dict(item)
    result["security"] = capability_security_metadata(
        kind=kind,
        identifier=result.get(identifier_key),
        source_type=source_type,
    )
    return result
