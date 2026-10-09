"""Security contracts for tenant-scoped knowledge-base ingestion and retrieval."""

from types import SimpleNamespace

import pytest

from core.auth.kb_permissions import (
    get_accessible_local_kb_levels,
    resolve_user_kb_context,
)
from core.content.kb_processing import (
    build_document_security_metadata,
    document_security_summary,
    validate_indexing_identity,
)


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *_args):
        return self

    def all(self):
        return self.rows


class _DB:
    def __init__(self, users, spaces=()):
        self.users = users
        self.spaces = list(spaces)

    def get(self, _model, key):
        metadata = self.users.get(key)
        return SimpleNamespace(extra_data=metadata) if metadata is not None else None

    def query(self, _model):
        return _Query(self.spaces)


def _space(kb_id, owner, visibility, tenant_id=None, authoritative=True):
    metadata = {"tenant_id": tenant_id} if tenant_id else {}
    if tenant_id and authoritative:
        metadata["tenant_binding"] = {
            "version": 1,
            "source": "authoritative_user_context",
        }
    return SimpleNamespace(
        kb_id=kb_id,
        user_id=owner,
        visibility=visibility,
        extra_data=metadata,
    )


def test_kb_access_is_tenant_scoped_and_fails_closed_without_role():
    users = {
        "user-a": {"tenant_id": "tenant-a", "role": "member"},
        "admin-a": {"tenant_id": "tenant-a", "role": "super_admin"},
        "owner-a": {"tenant_id": "tenant-a", "role": "member"},
        "owner-b": {"tenant_id": "tenant-b", "role": "member"},
        "missing-role": {"tenant_id": "tenant-a"},
    }
    db = _DB(
        users,
        [
            _space("kb-own", "user-a", "private", "tenant-a"),
            _space("kb-shared-a", "owner-a", "public", "tenant-a"),
            _space("kb-shared-b", "owner-b", "public", "tenant-b"),
            _space("kb-spoof", "owner-b", "public", "tenant-a", authoritative=False),
            _space("kb-unbound", "system", "public"),
        ],
    )

    assert get_accessible_local_kb_levels(db, "user-a") == {
        "kb-own": "admin",
        "kb-shared-a": "view",
    }
    assert get_accessible_local_kb_levels(db, "admin-a") == {
        "kb-own": "admin",
        "kb-shared-a": "admin",
    }
    assert resolve_user_kb_context(db, "missing-role") is None
    assert get_accessible_local_kb_levels(db, "missing-role") == {}


def test_document_security_metadata_keeps_authority_outside_user_metadata():
    metadata = build_document_security_metadata(
        tenant_id="tenant-a",
        user_id="user-a",
        kb_id="kb-1",
        document_id="doc-1",
        source_type="direct_upload",
        checksum_sha256="a" * 64,
        content_check={
            "status": "partial",
            "file_type_validation": "passed",
            "malware_scan": "not_run",
            "prompt_injection_scan": "not_run",
        },
        user_metadata={
            "tenant_id": "tenant-b",
            "indexing_identity": {"principal_id": "attacker"},
        },
    )

    assert metadata["tenant_id"] == "tenant-a"
    assert metadata["indexing_identity"]["principal_id"] == "user-a"
    assert metadata["user_metadata"]["tenant_id"] == "tenant-b"
    validate_indexing_identity(
        metadata,
        tenant_id="tenant-a",
        user_id="user-a",
        kb_id="kb-1",
        document_id="doc-1",
    )
    with pytest.raises(PermissionError, match="indexing identity mismatch"):
        validate_indexing_identity(
            metadata,
            tenant_id="tenant-b",
            user_id="user-a",
            kb_id="kb-1",
            document_id="doc-1",
        )


def test_document_security_summary_is_redacted():
    metadata = build_document_security_metadata(
        tenant_id="tenant-a",
        user_id="user-a",
        kb_id="kb-1",
        document_id="doc-1",
        source_type="direct_upload",
        checksum_sha256="a" * 64,
        content_check={
            "status": "partial",
            "file_type_validation": "passed",
            "malware_scan": "not_run",
            "prompt_injection_scan": "not_run",
        },
    )
    summary = document_security_summary(
        SimpleNamespace(
            kb_id="kb-1",
            document_id="doc-1",
            extra_data=metadata,
        )
    )

    assert summary["tenant_bound"] is True
    assert summary["indexing_identity"] == {"bound": True}
    assert summary["provenance"]["source_type"] == "direct_upload"
    assert "tenant-a" not in str(summary)
    assert "user-a" not in str(summary)
